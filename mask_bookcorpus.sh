mkdir bookcorpus
cd bookcorpus
wget https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2
tar -jxvf bookcorpus.tar.bz2
awk '/isbn/ {n++} {print > "book" n ".txt"}' books_large_p1.txt
rm books_large_p1.txt
awk 'BEGIN {n=479} /isbn/ {n++} {print > "book" n ".txt"}' books_large_p2.txt
rm books_large_p2.txt