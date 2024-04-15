from pydub import AudioSegment
from pydub.silence import split_on_silence

# 入力と出力ファイル名
input_file = "input.m4a"
output_file = "output.m4a"

# 音声ファイルを読み込み
sound = AudioSegment.from_file(input_file, format="m4a")

# 無音部分を検出して分割
chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40)

# 無音部分で分割された音声チャンクを連結
output = chunks[0]
for chunk in chunks[1:]:
    output += chunk

# 分割された音声を結合して出力
output.export(output_file, format="m4a")

