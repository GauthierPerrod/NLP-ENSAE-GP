cat cooking.stackexchange.test.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.stackexchange.test.preprocessed.txt

cat cooking.stackexchange.train.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.stackexchange.train.preprocessed.txt

cat cooking.stackexchange.valid.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.stackexchange.valid.preprocessed.txt

cat langid.test.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > langid.test.preprocessed.txt

cat langid.train.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > langid.train.preprocessed.txt

cat langid.valid.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > langid.valid.preprocessed.txts
