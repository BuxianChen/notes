## 音频格式

无损格式：wav, flac
有损格式：mp3

wav格式为 44 字节作为头部，后续字节为实际的数据。一般采用 int16(取值范围是`[-32768, 32767]`) 来存储。因此在事先已知音频的“元数据”信息时，可以以下面的方式推算音频时长：

```python
import os
wav_path = "x.wav"
sr = 16000  # 原始数据的sample_rate为16000
n_channels = 1  # 通道数
size = 2  # 单个采样点所占的字节数(int16为2个字节)
seconds = (os.path.getsize(wav_path) - 44) / 2 / n_channels / sr
```

## Python读写语音的包

- scipy.io.wavfile
- soundfile: [doc](https://pysoundfile.readthedocs.io/en/latest)
- audioread
- librosa
- torchaudio
- wavefile
- wave

依赖关系及安装：
（1）torchaudio (0.11.0) 当前可以选用 soundfile 与 sox_io 作为 backend
（2）librosa (v0.7以后) 使用 soundfile 与 audioread 作为 backend 来读写音频。特别的，默认使用 soundfile 进行读写，特别地：mp3 格式的文件 soundfile 无法读取，librosa 会用 audioread 进行读写。
（3）soundfile 的安装步骤为：
```bash
apt install libsndfile1
pip install soundfile
```
（4）audioread 的安装步骤为：
```bash
# apt install ffmpeg  # 有些系统必须先安装ffmpeg
pip install audioread
```

### scipy.io.wavfile

只能读写 wav 格式，且 format 只能是 `32-bit floating-point`，`32-bit PCM`，`16-bit PCM`，`8-bit PCM`

```python
from scipy.io import wavfile
path = "x.wav"
# rate为采样率(int), 多通道情况下x为的形状为:(Nsamples, Nchannels), 单通道时为:(Nsamples,)
# 读取的数据不会做任何归一化
rate, x = wavfile.read(path)
```

注意：写文件时需要先将对音频resample好，才能写入

```python
from scipy.io import wavfile
path = "x.wav"
rate, x = wavfile.read(path) # rate=8000, len(x)=16000
wavfile.write("y.wav", 16000, x)  # 错误, 需要预先将音频resample好
rate, y = wavfile.read("y.wav")  # rate=16000, len(x)=16000
```

### soundfile

按[官方文档](https://pysoundfile.readthedocs.io/en/latest)的说法, soundfile的API在0.6,0.7,0.8发生了些变化，需要小心这些“坑”。

```python
path = "x.wav"
# 参数很多, 此处不列举全
x, sr = soundfile.read(path, dtype='float64', always_2d=False)
# 多通道情况下, x的形状为(Nsamples, Nchannels)
# 通常情况下, x将被归一化至[-1, 1), 例如原始文件里村存的是int16, 而read函数的参数为"float64", 归一化方式为除以2^15=32768。但如果原始数据按float方式存，但读取时按int来读，则不会做归一化
```
soundfile 还提供了一些底层API例如：
```python
import soundfile as sf

with sf.SoundFile('myfile.wav', 'r+') as f:
    while f.tell() < f.frames:
        pos = f.tell()
        data = f.read(1024)
        f.seek(pos)
        f.write(data*2)
```

### audioread

略

### librosa

```
```

librosa读取数据与soundfile一样一般会做归一化，但多通道情况下，读取出来的维数顺序会与soundfile相反，原因在于如下源码：

```python
# librosa.read源码
y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T
```

### wave

```python
import wave
import contextlib
with contextlib.closing(wave.open(path, "rb")) as wf:
    num_channels = wf.getnchannels()  # 获取通道数
    sample_width = wf.getsampwidth()  # 每个sample需要的字节数, 例如通常用2个字节存储一个采样点
    sample_rate = wf.getframerate()  # 采样率(1s钟多少个采样点)
    nframes = wf.getnframes()  # 采样点数, 即采样率乘以秒数
    pcm_data = wf.readframes(nframes)  # 字节
# 对于单通道数据
import struct
# <表示little-endian, 2表示读两个数字, H表示数据类型为uint16
struct.unpack("<2H", pcm_data[:4])  # 获取前两个采样点的值
```