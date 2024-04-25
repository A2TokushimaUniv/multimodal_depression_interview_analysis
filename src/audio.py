import opensmile


# TODO: OpenSmile以外にSurfboardやlibrosa、HuBERTなどを試してみる
def get_opensmile_feature(audio_file):
    # https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_1O3GS705/_pdf/-char/en では両方使っている
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feature = smile.process_file(audio_file)
    return feature
