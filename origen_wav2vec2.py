import torch  #PyTorch関係
import torch.nn.functional as F  #テンソル計算 GPU計算 F:ニューラルネットワークの関数群
import torch.optim as optim #Adam、SGDなどの最適化アルゴリズム
from warnings import simplefilter  #警告メッセージを抑制する
import argparse  #コマンドライン引数を扱う コマンドライン引数を使って、実行できるようにするため
from tqdm import tqdm  #進捗バーの表示
from pathlib import Path
import numpy as np  #配列演算
import scipy  #信号処理 ex)scipy.signalでフィルタ処理
import librosa  #音声解析ライブラリ 音声の読み込みを行う
import random  #乱数
import imageio #画像保存 攻撃途中のスペクトログラムの可視化に使われることが多い
import wave  #wavファイルの読み書き
from typing import Union  #型ヒント用
import os
import glob
import math 
import json
import matplotlib.pyplot as plt  #攻撃損失の推移を描画
import sys
#from transformers import Wav2Vec2Model

#各クラウド音声認識を使う関数を読み込み
from cloud_decode.aliyun_function import *
from cloud_decode.tencentyun_function import *
from cloud_decode.google_api import *
from cloud_decode.azure_api import *
from cloud_decode.openai_function import openai_recong

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #GPU0のみ利用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #GPUがなければCPU

simplefilter(action='ignore', category=Warning)  #大量の警告を表示しないため
parser = argparse.ArgumentParser(description='ASR attack')  #コマンドライン引数の定義を開始
plt.rcParams.update({'font.size': 24})  #グラフのフォントサイズ
torch.backends.cudnn.enabled = False  #cuDNNを無効化、高速化ライブラリを無効化することで実験結果が安定する

#引数定義
parser.add_argument('--seed', type=int, default=2025, metavar='S',help='random seed (default: 2023)')  #乱数シード2025に固定
parser.add_argument('--epoch', type=int, default=1500)  #最適化回数。敵対的摂動を1500回
parser.add_argument('--speech-file-path', default='',type=str, help='speech file path')  #元音声wav
parser.add_argument('--music-file-path', default='',type=str, help='music file path')  #背景音楽
parser.add_argument('--attack-target', default='aliyun',type=str,choices=['tencentyun','aliyun','baiduyun','iflytec','google','azure','openai'])  #攻撃対象のASR選択肢から指定
parser.add_argument('--sample-num', default=0,type=int, help='samples num')  #評価するサンプル番号、データセット管理用
parser.add_argument('--sound-db', default=75,type=int, help='sound db')  #音圧レベル、絶対音の大きさを制御
parser.add_argument('--start-second', type=float, default=0.5, help='start second in music to add perturbation')  #背景音楽のどこから攻撃音を埋め込むか

class Attacker:
    def __init__(self, args):  #コンストラクタ、実行時のパラメータを受けとる
        self.args = args  #パラメータ保存
        self.epoch=args.epoch  #パラメータ保存
        self.sound_db = args.sound_db  #パラメータ保存
        self.start_second = args.start_second  #パラメータ保存
        #STFTパラメータ、音声をスペクトログラム化するための設定
        self.n_fft=512  #FFE窓サイズ16㎑音声なので、512／16000ごとに周波数解析
        self.hop_length=256  #フレーム移動量
        self.frame_num=240  #使うフレーム数
        self.seq_len=self.hop_length*(self.frame_num-1)+self.n_fft  #音声長の計算 計算すると 61696samples 最大3.856秒になる

        #攻撃対象ASRの決定
        if args.attack_target == 'tencentyun':
            self.decode_function=tencent_recong
        if args.attack_target == 'aliyun':
            self.decode_function=aliyun_recong
        if args.attack_target == 'google':
            self.decode_function=google_decode
        if args.attack_target == 'azure':
            self.decode_function=azure_decode
        if args.attack_target == 'openai':
            self.decode_function=openai_recong

        # 正規化関数
        self.normalize=normalize_e
        self.denormalize=denormalize_e

        # 攻撃対象音声の読み込み
        self.speech,_=self.read_wav_file(args.speech_file_path)  #音声の読み込み
        speech_name=args.speech_file_path.split('/')[-1].split('.')[0]  #音声のファイル名を区切って単語にする(speech_name)
        self.command=speech_name.replace('_',' ')  #保存先生成
        physical_samples_path=Path.home()/('physical_samples')/speech_name/args.attack_target  #ここのファイルに保存する
        if not os.path.exists(physical_samples_path):
            os.makedirs(physical_samples_path)  #存在しなければファイルを作成する
        self.physical_samples_path=physical_samples_path

        self.decode_count=0  #音声認識APIを何回読んだか数えるため（ブラックボックス攻撃では推論回数＝コストなので重要）
        music_file_list=glob.glob(args.music_file_path+'/*.wav')  #wavのファイルを所得
        print(f"Found {len(music_file_list)} music files to attack.")  #件数表示

        for music_file in music_file_list:
            # 音楽音声の読み込み
            music_name=music_file.split('/')[-1].split('.')[0]  #サンプル名生成
            music,start_time=self.read_wav_file(music_file)  #読み込み開始
            self.attack_music_name=speech_name+'_'+music_name+'_'+str(start_time)  #名前の生成
            print(self.attack_music_name,'  attack begin!!')
            success=self.distribution_attack(self.speech,music)  #ここが本体
            if success:
                print(self.attack_music_name,'  Success!!')  #成功
            else:
                print(self.attack_music_name,'  Failed!!')  #失敗
                
    #命令音声の読み込みと音量調整
    def read_wav_file(self,file_path,start_time=0):
        wav_data,_ = librosa.load(file_path, sr=16000, mono=True)  #読み込み16kHz、モノラルに変換
        if len(wav_data)>self.seq_len:  #長い場合は切り出し 音声の長さの最大は3.86秒
            if not start_time:
                start_time=0  #スタートタイムを0にする
            wav_data=wav_data[start_time:start_time+self.seq_len]  #長い音声はここで短縮される（データの長さを0～3.856秒）
        else:
            wav_data=wav_data[np.convolve(np.abs(wav_data), np.ones((512))/512, mode='same')>0.001]  #無音を除去（波形の絶対値とる、長さ512の平均化フィルタを作る、移動平均をかける、音量が0.001以下のところはTrue→無音削除、音がある部分だけ残す）
        mean_amp = np.mean(np.abs(wav_data*(2**15)))  #振幅の平均を求める→音の平均的な大きさが分かる
        scale_ratio = self.compute_scale_ratio(mean_amp, self.sound_db, max_amp_value=32768.)  #平均振幅を既定（75dB）に合わせる倍率を計算
        wav_data = scale_ratio * wav_data  #求めた倍率を掛けて、音量を調節（音量を一定にする）ここでさっきのを適応する
        wav_data=np.clip(np.asarray(wav_data).ravel(),-1,1).astype(np.float32)  #波形を一次元化、音割れを防ぐ(ここに修正あり)
        return wav_data,start_time  #整形済み波形と切り出し位置を返す

    def compute_scale_ratio(self,amp, expected_db, max_amp_value: float = 32768.):  #音量倍率の計算
        return np.power(10, (expected_db + 20 * np.log10(max_amp_value) - 96.) / 20) / amp  #平均振幅を目標のdBに近づける倍率を計算する ＊倍率を返す＊ ampが小さいほど大きな倍率になる
        # expected→目標の音量、20 * np.log10(max_amp_value)→最大振幅を㏈変換、96→㏈FS補正。最大振幅の基準に合わせるために96㏈引く　目標dB（75dB）にするために何倍すればいいのか
    
    # 命令音声と元音声から最適化と候補音声生成を行う
    def distribution_attack(self,speech,music): 
        music_mag,music_stft_phase=wav_preprocess(music,self.n_fft,self.hop_length)  #元の音楽をSTFT変換 振幅と位相スペクトグラムに分ける（wav_preprocessで振幅と位相スペクトグラムを返す）
        music_mag=self.normalize(music_mag)  #振幅にlogを正規化、最適化を安定させる（周波数×時間）
        clean_music=conc_tog_specphase(music_mag.unsqueeze(0),music_stft_phase.unsqueeze(0),n_fft=self.n_fft,hop_length=self.hop_length).squeeze()  #正規化済みの振幅をもとに戻して元の位相と合成して音楽を再構成
        speech_mag,_=wav_preprocess(speech,self.n_fft,self.hop_length)  #命令音声もSTFT変換 振幅と位相に分ける　位相スペクトグラムは捨てる
        speech_mag=self.normalize(speech_mag)  #同様に振幅をlogに正規化
        speech_frame=speech_mag.size(1)  #命令音声の時間フレーム数を取得、どの区間へ入れるかを決めるため。命令音声のフレームの領域を探す
        speech_rhythm=F.softmax(torch.mean(square_smooth(self.denormalize(speech_mag),square_kernel_size=[7]),dim=0))  #命令音声の振幅変化から、時間方向の強さの分布を作る。ここでは使っていない


        smooth_music_mag=square_smooth(music_mag,kernel_size=[15])  #音楽スペクトグラムの15×15の平均フィルタで平滑化
        seq_idx = 0
        frame_idx = int(self.start_second * 16000 / self.hop_length)  #スタートセカンドをフレーム番号へ変換、1フレームは約16ms
        
        # Ensure frame_idx does not exceed total music frames
        max_frame_idx = music_mag.size(1) - speech_mag.size(1)  #命令音声を収められる音楽側の最大開始フレームを計算
        if frame_idx > max_frame_idx:  #指定開始位置が範囲外なら、収まる最大位置へ補正
            print(f"Warning: start_second={self.start_second}s exceeds music length, using max possible frame_idx={max_frame_idx}")
            frame_idx = max(0, max_frame_idx)
        
        #print('frame_idx:', frame_idx, '    speech_frame:', speech_frame.item())

        smooth_speech_mag=square_smooth(speech_mag,kernel_size=[15])   #命令音声のスペクトグラムの平滑化
        speech_feature=(smooth_speech_mag-square_smooth(smooth_speech_mag,kernel_size=[15]))  #一度平滑化した結果からさらに平滑化した結果を引く、音声の局所的な特徴を取り出すためである
        music_feature=(smooth_music_mag-square_smooth(smooth_music_mag,kernel_size=[15]))[:,frame_idx:frame_idx+speech_frame]  #音楽側でも同様、命令音声と同じ長さ、指定位置の範囲にを切り出す
        feature_mask=square_smooth(torch.tensor(square_smooth(torch.abs(speech_feature),kernel_size=[7])<0.005).float(),kernel_size=[5]) #命令音声の特徴量が小さい領域を判定し、マスクを作る
        #target_feature=speech_feature+feature_mask*music_feature

                # Create frequency mask (200Hz - 7000Hz)
        sr = 16000  #サンプルレートを16kHzとして設定
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)  #FTTの各周波数が何Ｈｚに対応するのか配列で取得
        freq_mask = (freqs >= 200) & (freqs <= 7000)  #200～7000ＨｚだけTureにするマスクを作る
        # Convert to Tensor and adjust shape for broadcasting
        self.freq_mask_tensor = torch.tensor(freq_mask, dtype=torch.float32, device=device).unsqueeze(1)  #マスクをPyTorchテンソル化



        advised_gain=torch.zeros_like(music_mag)  #敵対的な変更量を用意する/元音楽のスペクトグラムと同じ型の変更量作成/縦軸は周波数、横軸は時間
        #advised_gain = 1e-3 * torch.rand_like(music_mag);
        advised_gain.requires_grad=True  #この変更量を微分の対象にする/損失を小さくするためにどこをどれだけ変えるのか計算
        optimizer=optim.Adam(params=[advised_gain],lr=4e-2)  #更新する最適化器を生成/lr=4e-2は一回の更新幅
        balance_param=30  #損失のバランス調整用変数
        #ここから学習の繰り返し
        with tqdm(total=self.epoch, desc='Adversarial Train') as train_enum:  #学習回数と進捗を表示する
            for epoch in range(self.epoch):  #指定回数更新を繰り返す
                with torch.autograd.set_detect_anomaly(True):  #異常なエラーが起きた時に場所を検出しやすくする
                    # Smooth perturbation and enforce non-negative (use softplus to maintain differentiability), ensuring perturbation is always greater than zero
                    raw_smooth_advised = square_smooth(advised_gain, square_kernel_size=[3])  #学習中の変更量を3×3の平均化で滑らかにする
                    # Use softplus to map arbitrary real numbers to positive values, avoiding negative perturbations
                    smooth_advised_gain = F.softplus(raw_smooth_advised) * self.freq_mask_tensor  #周波数マスクをかけて、200～7000Hz以外の変更量を0にする
                    #smooth_advised_gain = F.softplus(raw_smooth_advised)
                    advised_mag=torch.clamp_min(music_mag+smooth_advised_gain,0)  #元音楽の振幅スペクトグラムに変化量を足す（負にならないように）
                    
                    #スペクトグラムを音声に戻し、対象区間を取り出す
                    start_sample = frame_idx * self.hop_length  #挿入開始フレームと命令音声の長さから、開始終了サンプル位置を計算
                    end_sample = (frame_idx + speech_frame) * self.hop_length + self.n_fft
                    full_advised_audio = conc_tog_specphase(advised_mag.unsqueeze(0), music_stft_phase.unsqueeze(0), 
                                                               n_fft=self.n_fft, hop_length=self.hop_length).squeeze()  #加工後の振幅と元音声の位相から加工後の音声波形を復元
                        
                    advised_audio = full_advised_audio[start_sample:end_sample]  #復元した音声のうち、音声命令に対応する区間だけ取り出す
                    original_music = clean_music[start_sample:end_sample]  #同じ区間の加工前の元音楽も取り出す
                    
                    # 命令音声をPyTorchでテンソル変換/CPUへ配置
                    speech_tensor = torch.tensor(speech, dtype=torch.float32).to(device)
                    
                    # Ensure consistent length (take the shorter length)
                    min_length = min(speech_tensor.shape[0], original_music.shape[0], advised_audio.shape[0])  #命令音声、元音楽、加工後音声の長さのうち、最短の長さを求める
                    speech_tensor = speech_tensor[:min_length]
                    original_music = original_music[:min_length]
                    advised_audio = advised_audio[:min_length]  #3つの音声を同じ長さに切り詰める
                    
                    # Target = command + corresponding part of original music
                    target_audio = speech_tensor  #加工後の音楽を近づけたい目標音声を、命令音声そのものに設定

                    # Log-Mel特徴量と損失を計算する
                    target_feat = compute_log_mel_spectrogram(target_audio, n_fft=self.n_fft,   #命令音声を80次元のLog-Melスペクトグラムへ変換。これが近づけたい特徴量
                                                                    hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=True)
                    advised_feat = compute_log_mel_spectrogram(advised_audio, n_fft=self.n_fft,   #加工後の音楽についても同様
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=True)
                    advised_mel=compute_log_mel_spectrogram(advised_audio, n_fft=self.n_fft,  #加工後の音楽の対数化前のMelスペクトグラムを計算
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)
                    original_music_log_mel = compute_log_mel_spectrogram(original_music, n_fft=self.n_fft,  #加工前音楽のLog‐Mel特徴量
                                                                            hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=True)
                    original_music_mel=compute_log_mel_spectrogram(original_music, n_fft=self.n_fft,  #加工前音楽のMelスペクトグラム
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)
                    # NaNや無限大のとき、更新を中止
                    if not torch.isfinite(target_feat).all() or not torch.isfinite(advised_feat).all():
                        print("Warning: NaN or Inf detected in log-mel spectrograms, skipping this iteration")
                        continue
                    
                    # Ensure both log-Mel spectrograms have consistent time dimensions
                    min_time_frames = min(target_feat.shape[1], advised_feat.shape[1])  #目標と加工後で、時間フレーム数の短い方を取得
                    target_feat = target_feat[:, :min_time_frames]
                    advised_feat = advised_feat[:, :min_time_frames]
                    original_music_log_mel = original_music_log_mel[:, :min_time_frames]
                    advised_mel = advised_mel[:, :min_time_frames]
                    original_music_mel = original_music_mel[:, :min_time_frames]  #各特徴量の時間長江尾そろえる
                    # Calculate MAE difference for entire log-Mel spectrogram
                    #ここの特徴量をwav2vac2に変更する
                    wav2vec2_loss = F.l1_loss(advised_feat, target_feat)  #加工後音楽と命令音声のLog‐Mel特徴量の平均絶対誤差を求める。小さいほど両者の特量が近くなる

                    # Calculate advised_feat / advised_mel directly from spectrum amplitude (no need to convert to audio first)
                    seg_len = int(speech_frame.item()) if torch.is_tensor(speech_frame) else int(speech_frame)
                    # Calculate 80-dimensional log-Mel spectrogram (apply CMVN normalization) - only for noise loss calculation
                    advised_mel=compute_log_mel_spectrogram(advised_audio, n_fft=self.n_fft,
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)
                    
                    original_music_mel=compute_log_mel_spectrogram(original_music, n_fft=self.n_fft,
                                                                 hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)

                
                    seg_len = int(speech_frame.item()) if torch.is_tensor(speech_frame) else int(speech_frame)
                    seg_start = frame_idx
                    seg_end = frame_idx + seg_len
                    T_time = music_mag.shape[1]
                    if seg_start >= T_time:
                        continue
                    seg_end = min(seg_end, T_time)
                    # Clean music spectrum (linear amplitude) in this time segment
                    # clean_mag_linear = self.denormalize(music_mag)
                    # clean_seg = clean_mag_linear[:, seg_start:seg_end]
                    # # Noise gain in this time segment
                    # noise_seg = smooth_advised_gain[:, seg_start:seg_end]
                    
                    # Calculate NSR (Noise-to-Signal Ratio) in 50ms segments
                    # Number of frames for 50ms: 50ms / (hop_length/sr * 1000) = 50 / (258/16000 * 1000) ≈ 3.1 frames
                    # frames_per_50ms = max(1, int(0.05 * 16000 / self.hop_length))  # Number of frames for 50ms
                    # seg_length = clean_seg.shape[1]
                    # nsr_list = []
                    
                    # for start_idx in range(0, seg_length, frames_per_50ms):
                    #     end_idx = min(start_idx + frames_per_50ms, seg_length)
                    #     # Extract noise and signal for 50ms segment
                    #     noise_segment = noise_seg[:, start_idx:end_idx]
                    #     clean_segment = clean_seg[:, start_idx:end_idx]
                    #     # Calculate energy for this segment
                    #     noise_energy = torch.sum(noise_segment ** 2) + 1e-10
                    #     clean_energy = torch.sum(clean_segment ** 2) + 1e-10
                    #     # NSR = noise energy / signal energy
                    #     nsr = noise_energy / clean_energy
                    #     nsr_list.append(nsr)
                    
                    # # Average NSR across all 50ms segments
                    # noise_loss = torch.mean(torch.stack(nsr_list)) if nsr_list else torch.tensor(0.0, device=device)
                    
                
                    # 最終損失と更新
                    if not torch.isfinite(wav2vec2_loss):
                        print("Warning: wav2vec2_loss is not finite, using zero loss")
                        wav2vec2_loss = torch.tensor(0.0, device=device, requires_grad=True)  #ここを変える！
                    
                    # frame_snrs = []
                    # for frame_idx in range(advised_mel.shape[1]):  # Iterate over each time frame
                    #     # Extract signal and noise for current frame
                    #     frame_signal = original_music_mel[:, frame_idx]  # shape: [n_mels]
                    #     frame_noise = advised_mel[:, frame_idx] - frame_signal
                    #     # Calculate SNR for current frame
                    #     frame_snr = torch.sum(frame_noise ** 2) / (torch.sum(frame_signal ** 2) + 1e-8)
                    #     if torch.isfinite(frame_snr):
                    #         frame_snrs.append(frame_snr)
                    # noise_loss = torch.mean(torch.stack(frame_snrs))
                    #loss=wav2vec2_loss+40*noise_loss
                    #loss = wav2vec2_loss+0.015*torch.norm(smooth_advised_gain,p=2)
                    #Match music? loss=wav2vec2_loss+0.4*F.l1_loss(advised_feat,original_music_log_mel)
                    #loss=wav2vec2_loss+0.30*torch.norm(advised_mel-original_music_mel,p=2)
                    #loss = diff_loss_freq+F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))
                    loss = wav2vec2_loss+200*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))  #最終損失を生成
                    #loss = wav2vec2_loss+50*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))
                    #noise_loss=F.mse_loss(advised_mel/ (original_music_mel + 1e-8),torch.ones_like(original_music_mel))
                    #loss=wav2vec2_loss+0.7*noise_loss#0.8 whisper
                    #noise_loss=F.l1_loss(advised_feat/ (original_music_log_mel + 1e-8),torch.ones_like(original_music_log_mel))
                    #√ loss=wav2vec2_loss+0.4*noise_loss
                    #loss=wav2vec2_loss+30*noise_loss
                    #loss = wav2vec2_loss+400*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))+10*noise_loss
                    #loss = wav2vec2_loss+200*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))+0.01*torch.norm(advised_mel-original_music_mel,p=2)
                    #loss = wav2vec2_loss+600*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))+5*noise_loss
                    # Ensure loss is finite
                    if not torch.isfinite(loss):
                        print("Warning: total loss is not finite, skipping backward pass")
                        continue
                        
                    
                    optimizer.zero_grad()  #損失から逆向きに計算し、advised_gainの各要素をどちらへ変えるべきかという勾配を求める
                    loss.backward(retain_graph=True)
                    
                    # Check gradients
                    if torch.isfinite(advised_gain.grad).all():
                        optimizer.step()  #勾配を使って更新する。ここで加工内容が変化する
                    else:
                        print("Warning: gradients contain NaN or Inf, skipping optimizer step")
                        
                    train_enum.set_description(f'Train attacker:(loss:{loss:.10f},wav2vec2_loss:{wav2vec2_loss:.8f})')  #進捗バーに表示
                    train_enum.update()  #進捗バーを1回進める

                #torch.cat((interpolate_smooth_speech_mag,speech_feature,feature_mask,advised_mag,music_mag),1).cpu().detach().numpy()
        
        # -------学習後のマスク処理---------
        #mask=torch.zeros_like(music_mag)  #元音楽のスペクトグラムと同じ形で、全要素が0のマスクを作る
        #mask[5:,frame_idx-2:frame_idx+speech_frame+2]=1  #低周波ビンを避けつつ、命令音声に対応する時間範囲だけを1にする
        #mask=square_smooth(mask,square_kernel_size=[5])  #マスクの境界を滑らかにする
        #feature_mask=square_smooth(torch.tensor(square_smooth(torch.abs(speech_feature),kernel_size=[7])>0.001).float(),kernel_size=[5])  #命令音声のスペクトル特徴が一定以上ある箇所を1に近づけるマスクを作る
        #mask[:,frame_idx:frame_idx+speech_frame]*=feature_mask  #大まかな時間、周波数マスクに命令音声由来の特徴マスクをかける

        #perturbation = smooth_advised_gain * mask
        #print("maskの最大値:", mask.max().item())
        #print("feature_maskの最大値:", feature_mask.max().item())
        #print("smooth_advised_gainの最大値:", smooth_advised_gain.max().item())
        #print("摂動の最大値:", perturbation.abs().max().item())
        #print("摂動が0でない要素数:", (perturbation != 0).sum().item())
        #advised_mag = torch.clamp_min(music_mag + perturbation, 0)

        #advised_mag=torch.clamp_min(music_mag+(smooth_advised_gain)*mask,0)  #学習済みの変更量に最終マスクをかけ、元音楽へ加える。これが最終的な加工後スペクトグラム

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        # 学習後のマスク処理
        mask = torch.zeros_like(music_mag)

        mask_start = max(0, frame_idx - 2)
        mask_end = min(music_mag.size(1), frame_idx + speech_frame + 2)

        #print("music_magの形:", tuple(music_mag.shape))
        #print("frame_idx:", frame_idx)
        #print("speech_frame:", speech_frame)
        #print("mask範囲:", mask_start, mask_end)

        mask[5:, mask_start:mask_end] = 1
        #print("代入直後のmask最大値:", mask.max().item())

        mask = square_smooth(mask, square_kernel_size=[5])
        #print("平滑化後のmask最大値:", mask.max().item())

        feature_mask = square_smooth(
            torch.tensor(square_smooth(torch.abs(speech_feature), kernel_size=[7]) > 0.001).float(),
            kernel_size=[5],
        )

        mask[:, frame_idx:frame_idx + speech_frame] *= feature_mask

        perturbation = smooth_advised_gain * mask
        #print("feature_maskの最大値:", feature_mask.max().item())
        #print("smooth_advised_gainの最大値:", smooth_advised_gain.max().item())
        #print("摂動の最大値:", perturbation.abs().max().item())
        #print("摂動が0でない要素数:", (perturbation != 0).sum().item())

        advised_mag = torch.clamp_min(music_mag + perturbation, 0)

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        #imageio.imwrite('attack_stft.png',torch.cat((interpolate_smooth_speech_mag,speech_feature,feature_mask,advised_mag,music_mag),1).cpu().detach().numpy())
        #Griffin-Limアルゴリズムによる最終音声の復元
        wav=GLA(self.denormalize(advised_mag).unsqueeze(0),music_stft_phase.unsqueeze(0),self.n_fft,self.hop_length).squeeze()  #最終スペクトグラムをWAV波形に戻す
            
        audio_file = str(  #認証サービスに渡す一時的なWAVファイル名を生成
            self.physical_samples_path /
            f"{self.attack_music_name}_{self.args.attack_target}_candidate.wav"
        )
        wav_write(np.clip(wav.cpu().detach().numpy().ravel(),-0.9,0.9),audio_file,16000)  #復元した波形を0.9に制限し16㎑のWAVとして保存

        #--------AEの出力のみ行う--------
        print(f"AE candidate saved: {audio_file}")
        #return True
        #------------------------------

        #result,success = self.decode_function(audio_file)  #選択された認証サービスの関数を呼び、文字起こしする
        #self.decode_count+=1  #認証回数を1増やす
        #assert success,'Decode Failed!!!'  #通信失敗時
        #print("recognition answer:"+result)  #認証結果の文字列表示
        #print(self.physical_samples_path)  #成功音声を保存するフォルダを表示
        #time.sleep(20)
        #成功条件の判定
        #if self.command not in result.lower().replace('\'s',' is').replace(',','').replace('.','').replace('9','nine').replace('1','one'):  #命令ファイル名から作った文字列が認証結果に含まれるかを確認
        #    return False
        #else:  #命令文字列が含まれているとき、成功時の保存、品質評価の処理へ
            #成功時のSNR評価と保存
        clean_music_np = np.clip(clean_music.cpu().detach().numpy().ravel(),-0.9,0.9)  #加工前音楽をテンソルからNumPy配列へ戻し、振幅を制限
        wav_np = np.clip(wav.cpu().detach().numpy().ravel(),-0.9,0.9)  #加工後音楽も同じ形式へ変換
            
            # SNR評価
        noise_np = wav_np - clean_music_np  #加工前後の差分を変更量として扱う
        SNR=10*np.log10((np.sum(clean_music_np[:]**2))/(np.sum(noise_np**2)))  #元音楽のエネルギーを変更量のエネルギーで割って、dB値へ変換（値が大きいほど、変化量が元音楽に比べて小さいことを示す）
        seg_snr = self.calculate_seg_snr(clean_music_np, noise_np, sr=16000, segment_ms=50)  #50ms単位に区切った区間SNRを計算

            # Add ViSQOL score and segSNR to filename
        new_audio_filename = f"{self.attack_music_name}_{self.args.attack_target}_SNR{SNR:.3f}_segSNR{seg_snr:.3f}.wav"  #新しいファイル名を作成
        success_audio_file = str(self.physical_samples_path) + '/' + new_audio_filename
            
        wav_write(wav_np,success_audio_file,16000)    #成功した加工後の音声を保存
            # with open(success_audio_file.replace('.wav','.json'), 'w') as json_file:
            #     json.dump({'result':result}, json_file, ensure_ascii=False)
            #self.decode_count=0  #認識回数を0に戻す
        return True  #処理成功

    #加工前音声と変更量を比較し、短時間ごとのSNRを重み付き平均で求める関数
    def calculate_seg_snr(self, clean_audio, noise_audio, sr=16000, segment_ms=50):
        """
        Calculate weighted segmental SNR (Weighted Segmental SNR)
        Args:
            clean_audio: Clean audio (numpy array)
            noise_audio: Noise audio (numpy array)
            sr: Sample rate
            segment_ms: Segment length (milliseconds)
        Returns:
            weighted_seg_snr: Weighted segmental SNR value
        """
        # Ensure consistent length
        min_len = min(len(clean_audio), len(noise_audio))  #2つの配列を同じ長さにそろえる
        clean = clean_audio[:min_len]
        noise = noise_audio[:min_len]
        
        # Calculate samples per segment
        seg_len = int(sr * segment_ms / 1000)  #一区間のサンプル数を計算
        num_segments = min_len // seg_len  #何区間に分割できるか計算
        
        if num_segments == 0:  #50ms未満の音声なら、計算できないので0
            return 0.0
            
        segment_snrs = []
        segment_noise_energies = []
        
        for i in range(num_segments):  #各50ms区間を順番に処理
            start = i * seg_len  #現在の区間の開始、終了サンプル位置を決める
            end = start + seg_len
            
            seg_clean = clean[start:end]  #元音声と変更量の区間だけ取り出す
            seg_noise = noise[start:end]
            
            clean_energy = np.sum(seg_clean ** 2)  #それぞれの2乗和を計算し、エネルギーとする
            noise_energy = np.sum(seg_noise ** 2)
            
            # Avoid division by zero and log(0)
            if noise_energy < 1e-10:  #変更量がほぼゼロならSNRを非常に高い100dBとして扱う
                snr = 100.0 # Noise is extremely small, SNR is very high
                continue
            elif clean_energy < 1e-10:  #元音楽がほぼ無音なら、SNRを低い値として扱う
                snr = -100.0 # Signal is extremely small, SNR is very low
            else:
                snr = 10 * np.log10(clean_energy / noise_energy)  #通常に計算
                
            segment_snrs.append(snr)  #その区間のSNRと変更量エネルギーを記録する
            segment_noise_energies.append(noise_energy)
            
        # Calculate weighted average
        total_noise_energy = sum(segment_noise_energies)  #全区間の変更量エネルギーの合計
        if total_noise_energy < 1e-10:  #全体として変更量がほぼ0なら計算済みSNRの平均を返す
            return np.mean(segment_snrs)
            
        weights = [e / total_noise_energy for e in segment_noise_energies]  #各区間の変更量エネルギーが全体に占める割合を重みとして計算
        weighted_seg_snr = sum(w * s for w, s in zip(weights, segment_snrs))  #各区間のSNRを重み付きで平均（変更が大きいほど結果に強く影響）
        
        return weighted_seg_snr  #計算した区間SNRを返す


def apply_cmvn(log_mel, eps=1e-8):  #特徴量の正規化（Log-Mel特徴量を各周波数帯で平均0、分散1に揃えるためのもの）＊実際は使ってない
    """
    Apply CMVN (Cepstral Mean and Variance Normalization) to log-Mel spectrogram
    Args:
        log_mel: log-Mel spectrogram tensor, shape [n_mels, n_frames]
        eps: Small value to prevent division by zero
    Returns:
        normalized_log_mel: CMVN normalized log-Mel spectrogram
    """
    # Calculate mean and variance for each Mel band (along time dimension)
    mean = torch.mean(log_mel, dim=1, keepdim=True)  # shape: [n_mels, 1]/Mel帯域ごとに時間方向の平均値を計算
    var = torch.var(log_mel, dim=1, keepdim=True, unbiased=False)  # shape: [n_mels, 1]/Mel帯域ごとに時間方向の分散を計算
    
    # CMVN normalization: (x - mean) / sqrt(var + eps)
    normalized_log_mel = (log_mel - mean) / torch.sqrt(var + eps)  #各値から平均を引き、標準偏差で割る
    
    return normalized_log_mel #正規化済みのLog-Melを返す

# MFCCを作る関数（音声認識でよく使われる特徴量※このコードでは使わない）
def compute_mfcc_features(audio_tensor, n_fft=512, hop_length=258, n_mels=80, n_mfcc=13, sr=16000):
    """
    Compute MFCC features of audio
    Args:
        audio_tensor: Input audio tensor
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of Mel filters
        n_mfcc: Number of MFCC coefficients
        sr: Sample rate
    Returns:
        mfcc: MFCC feature tensor, shape [n_mfcc, n_frames]
    """
    # Keep tensor format to maintain gradient
    if isinstance(audio_tensor, torch.Tensor):
        audio_for_mfcc = audio_tensor
    else:
        audio_for_mfcc = torch.tensor(audio_tensor, dtype=torch.float32).to(device)
    
    # Calculate spectrogram using torch.stft, maintaining gradients
    stft = torch.stft(
        audio_for_mfcc, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=n_fft,
        window=torch.hann_window(n_fft).to(device),
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Calculate power spectrum
    power_spec = torch.abs(stft) ** 2
    
    # Create Mel filterbank
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=sr/2)
    mel_filters = torch.tensor(mel_filters, dtype=torch.float32).to(device)
    
    # Apply Mel filter
    mel_spec = torch.matmul(mel_filters, power_spec)
    
    # Ensure all values in mel_spec are positive, avoid log(0) or log(negative)
    mel_spec = torch.clamp_min(mel_spec, 1e-10)
    
    # Convert to log scale
    log_mel = torch.log(mel_spec)
    
    # Check and handle possible NaN or Inf values
    log_mel = torch.where(torch.isfinite(log_mel), log_mel, torch.zeros_like(log_mel))
    
    # Calculate DCT (Discrete Cosine Transform) to get MFCC
    # Create DCT matrix
    n_frames = log_mel.shape[1]
    dct_matrix = torch.zeros(n_mfcc, n_mels, dtype=torch.float32).to(device)
    
    for i in range(n_mfcc):
        for j in range(n_mels):
            dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_mels)
        if i == 0:
            dct_matrix[i, :] *= np.sqrt(1.0 / n_mels)
        else:
            dct_matrix[i, :] *= np.sqrt(2.0 / n_mels)
    
    # Apply DCT: MFCC = DCT_matrix @ log_mel
    mfcc = torch.matmul(dct_matrix, log_mel)
    
    return mfcc
# ここまでMFCCの話※使わない

#学習で使う特徴量の生成（Log-Mel特徴量の作成）
def compute_log_mel_spectrogram(audio_tensor, n_fft=512, hop_length=258, n_mels=80, sr=16000, apply_cmvn_norm=True):  #波形を80次元のMel特徴量へ変換する関数
    """
    Compute log-Mel spectrogram of audio
    Args:
        audio_tensor: Input audio tensor
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of Mel filters
        sr: Sample rate
        apply_cmvn_norm: Whether to apply CMVN normalization
    Returns:
        log_mel: log-Mel spectrogram tensor
    """
    # Keep tensor format to maintain gradient
    if isinstance(audio_tensor, torch.Tensor):  #入力がテンソルなら保持、NumPyならテンソル化して計算先へ
        audio_for_mel = audio_tensor
    else:
        audio_for_mel = torch.tensor(audio_tensor, dtype=torch.float32).to(device)
    
    # 波形をSTFTに変換。512サンプルの窓を使い、指定した区間で分析
    stft = torch.stft(
        audio_for_mel, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=n_fft,
        window=torch.hann_window(n_fft).to(device),
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Calculate power spectrum
    #power_spec = torch.abs(stft) ** 2  #各周波数、時刻のパワーを計算
    power_spec = (torch.abs(stft) ** 2).float()   # ★ ここを追加
    
    # Create Mel filterbank
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=sr/2)  #80本のMelフィルタを作成し、テンソル化
    mel_filters = torch.tensor(mel_filters, dtype=torch.float32).to(device)
    
    # Apply Mel filter
    mel_spec = torch.matmul(mel_filters, power_spec) #パワースペクトルをMelスペクトグラムへ変換
    
    # Ensure all values in mel_spec are positive, avoid log(0) or log(negative)
    mel_spec = torch.clamp_min(mel_spec, 1e-10)  #対数計算時の数値不安定を避けるための最小値を設定
    if not apply_cmvn_norm:  #Falseの場合は対数化しないMelスペクトグラムを返す
        return mel_spec
    # Convert to log scale, using torch.clamp to ensure numerical stability
    log_mel = torch.log(mel_spec)  #自然対数に変換
    
    # Check and handle possible NaN or Inf values
    log_mel = torch.where(torch.isfinite(log_mel), log_mel, torch.zeros_like(log_mel))  #NaN、無限大は0
    
    # Apply CMVN normalization
    #log_mel = apply_cmvn(log_mel)  #CMVNを適応（ここはコメントアウト）

    return log_mel  #Log-Mel特徴量を返す

# スペクトグラムの平滑化
def square_smooth(input,square_kernel_size=[],kernel_size=[]):  #周波数×時間の2次元データを移動平均で滑らかにする関数
    bias=torch.zeros(1).to(device)  #畳みこみ演算に使う、値0のバイアスを作る
    input=input.clone().unsqueeze(0).unsqueeze(0)  #元データを複製し、畳み込み用4次元「バッチ、チャネル、周波数、時間」にする
    for size in square_kernel_size:  #指定された正方形カーネルごとに平滑化
        kernel=(torch.ones(1,1,size,size)/size**2).to(device)  #値がすべて同じ正方形の平均化フィルタをつくる（size=3なら3×3）
        padding_size=int((size-1)/2)  #出力サイズと入力とほぼ同じに保つための余白サイズ
        input=F.pad(input,(padding_size,padding_size,padding_size,padding_size),mode='replicate')  #端の値を複製して周囲を埋め、端部でも平滑化できるように
        input=F.conv2d(input, kernel, bias, stride=1)   #2次元畳み込みを行い、周波数、時間の両方向で平均化
    for size in kernel_size:  #指定があれば周波数方向だけの平滑化
        kernel=(torch.ones(1,1,size,1)/size).to(device)  #縦長の平均化フィルタを生成。主に周波数方向で滑らかにする
        padding_size=int((size-1)/2)  #端を埋めてから畳み込み実行
        input=F.pad(input,(0,0,padding_size,padding_size),mode='replicate')
        input=F.conv2d(input, kernel, bias, stride=1)
    return input.squeeze()  #追加したバッチ・チャネル次元を取り除き、元と同じ2次元形状に戻す

#sobelフィルタでスペクトグラムの境界、急変部分を検出※呼び出しはなし
def edge_detection(input):
    bias=torch.zeros(1).to(device)
    input=input.clone().unsqueeze(0).unsqueeze(0)
    kernel=torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype=torch.float).to(device)
    padding_size=1
    input=F.pad(input,(padding_size,padding_size,padding_size,padding_size),mode='replicate')
    input=F.conv2d(input, kernel, bias, stride=1)
    return input.squeeze()

#入力全体を平均0、標準偏差1へ変更※呼び出しなし
def standardization(stft_distribution):
    return (stft_distribution-stft_distribution.mean())/stft_distribution.std()
    
# WAVファイルとして保存する
def wav_write(wav_signal: np.ndarray, wav_path: str, sample_rate: Union[int, float], scale_bit_length: bool = True, bit_length: int = 16):  #波形データをWAVファイルとして保存する関数
    """
    write the wav file
    :param wav_signal: wav signal.
    :param wav_path: wav path.
    :param scale_bit_length: whether scale signal to bit length.
    :param sample_rate: sample rate.
    :param bit_length: bit length.
    """
    output_dir = os.path.dirname(wav_path)  #保存先パスからフォルダ部分だけ取り出す
    if output_dir and not os.path.exists(output_dir):  #保存先フォルダが指定されていて、なければ作成
        os.makedirs(output_dir)
    wav_handler = wave.open(wav_path, "wb")  #WAVファイルをバイナリ書き込みモードで開く
    wav_handler.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed')) #音声形式を設定（1チャンネル、16bit、指定サンプルレート、非圧縮PCM）
    max_amp = 2 ** (bit_length - 1)  #16bit音声の最大振幅を計算
    if scale_bit_length:  #波形値を16bit整数の範囲へ拡大してint16に変換
        wav_data = (wav_signal * max_amp).astype(np.int16)
    else:
        wav_data = wav_signal.astype(np.int16)
    wav_data = np.clip(wav_data, -max_amp, max_amp - 1)  #保存可能な16bit整数範囲に収める
    wav_handler.writeframes(wav_data.tobytes())  #整数化した波形をWAVファイルに書き込む
    wav_handler.close()  #保存完了

#振幅をlog(振幅＋1)に変更する
def normalize_e(stft_mag):
    return torch.log(stft_mag+1)

#normalize_e()の逆変換を行う
def denormalize_e(stft_mag):
    return torch.pow(math.e,stft_mag)-1

#常用対数ベースの正規化※呼び出しなし
def normalize_10(stft_mag):
    return (torch.log10(torch.clamp_min(stft_mag,1e-10))+2.5)/7.5

#上記の逆変換※呼び出しなし
def denormalize_10(stft_mag):
    return torch.pow(10,stft_mag*7.5-2.5)

#　振幅と位相から波形を復元※学習の途中に現在の加工スペクトグラムが音声としてどのように聞こえる状態かを得るために使われる
def conc_tog_specphase(S, P,n_fft,hop_length):  #振幅と位相を組み合わせて、時間波形へ戻す関数
    S = denormalize_e(S) #Sは対数正規化された振幅なので、線形振幅に戻す
    P = P * np.pi  #πで割った形からラジアンにもどす
    SP = S * torch.complex(torch.cos(P),torch.sin(P))  #振幅と位相から複素スペクトグラムを復元
    wav = torch.istft(SP,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,length=hop_length*(S.size(2)-1)+n_fft) #逆SIFTを使い、複素スペクトグラムを時間波形に変換
    return wav  #復元した波形を返す

def wav_preprocess(wav,n_fft,hop_length):  #STFT処理/波形をSTFTでじかん、周波数表現へ変換する関数
    stft = torch.stft(torch.tensor(wav).to(device), n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,return_complex=False)  #波形を短時間フーリエ変換/結果には各フレームと各周波数便の複素数情報がはいる
    stft_mag=torch.abs(torch.sqrt(torch.sum(torch.pow(stft,2),dim=-1)+1e-10))  #実部^2+虚部^2の平方根を計算し、振幅を求める
    stft_phase=(torch.atan2(stft[:,:,1].data, stft[:,:,0].data)/np.pi)  #位相を計算/－1～1の範囲で保持
    return stft_mag,stft_phase  #振幅スペクトグラムと位相スペクトグラムを返す

# 最終波形を反復的に再構成
def GLA(S,P, n_fft, hop_length, n_iter = 1000):  #Griffin-Lim系の反復処理で、反復スペクトグラムからより整合的な波形を作る関数※規定で1000回
    P = P * np.pi  #位相を正規化済み値からラジアン変換
    for i in range(n_iter):  #指定回数だけ波形復元と位相更新を繰り返す
        SP = S * torch.complex(torch.cos(P),torch.sin(P))  #現在の振幅と位相から複素スペクトグラムを生成
        wav = torch.istft(SP,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,length=hop_length*(S.size(2)-1)+n_fft)  #複素スペクトグラムを逆STFTして、一度時間波形に戻す
        next_SP = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,return_complex=False)  #その波形を再びSTFTへ変換
        P = torch.atan2(next_SP[:,:,:,1].data, next_SP[:,:,:,0].data)  #再分析した複素スペクトグラムから位相を取り直すこれが次の反復で使う新しい位相
    return wav  #最後の反復で得た波形を返す

def main():
    args = parser.parse_args()
    Attacker(args)


if __name__ == '__main__':
    main()
