tail -n 19986 ../1_billion_word/1b-100k.dict > ../1_billion_word/1b-100k.embedding.dict

cat ../1_billion_word/1b-100k.embedding.dict | sed 's/[][()\.^$?*+]/\\&/g' | sed "s/^\(.*\)\s.*/\^\1\\\s/g" > queries.txt
cat ../1_billion_word/1b-100k.embedding.dict | cut -d' ' -f1 > query.txt

# does all the work
cat queries.txt | xargs -d'\n' -I{} rg -e {} wiki-news-300d-1M-subword.vec --no-line-number -m 1 > matches.txt

# figure out what we matched correctly
cat match.txt | cut -d' ' -f1 > match_names.txt

# <manually edit match.txt to include stuff we didn't match, _ doesn't exist but | does>
# top_words.vec is manually created since I was too aggressive about what .embedding.dict should be
python include_pipe.py

# now, just load the pytorch vector
python load_pytorch.py match_pipe.txt ../1_billion_word/1b-100k.dict
