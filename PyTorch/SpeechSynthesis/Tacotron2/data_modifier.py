# with open('$HOME/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/SLRDATA/line_index.tsv', 'r') as fr:
with open('filelists/bn_slr_SLR37_bn_bd_line_index.tsv', 'r', encoding='utf-8') as fr:
    for i, row in enumerate(fr):
        x = row.split('\t')
        newline = 'SLRDATA/wavs/' + x[0] + '.wav|' + x[1]
        if i < 1600:
            with open('filelists/bnslr_audio_text_train.txt', 'a+', encoding='utf-8') as fw:
                fw.writelines(newline)
        elif i >= 1600 and i<= 1800:
            with open('filelists/bnslr_audio_text_val.txt', 'a+', encoding='utf-8') as fw:
                fw.writelines(newline)
        else:
            with open('filelists/bnslr_audio_text_test.txt', 'a+', encoding='utf-8') as fw:
                fw.writelines(newline)