#!/usr/bin/env bash

set -e

DATADIR="SLRDATA"
FILELISTSDIR="filelists"

TESTLIST="$FILELISTSDIR/bnslr_audio_text_test.txt"
TRAINLIST="$FILELISTSDIR/bnslr_audio_text_train.txt"
VALLIST="$FILELISTSDIR/bnslr_audio_text_val.txt"

TESTLIST_MEL="$FILELISTSDIR/bnslr_mel_text_test.txt"
TRAINLIST_MEL="$FILELISTSDIR/bnslr_mel_text_train.txt"
VALLIST_MEL="$FILELISTSDIR/bnslr_mel_text_val.txt"

mkdir -p "$DATADIR/mels"
if [ $(ls $DATADIR/mels | wc -l) -ne 13100 ]; then
    python preprocess_audio2mel.py --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
    python preprocess_audio2mel.py --wav-files "$TESTLIST" --mel-files "$TESTLIST_MEL"
    python preprocess_audio2mel.py --wav-files "$VALLIST" --mel-files "$VALLIST_MEL"	
fi	