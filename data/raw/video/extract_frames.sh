rm -rf frames/* cropped/*
ffmpeg -r 1 -i $1 -r 1 "frames/%04d.bmp"
cd frames
mogrify -gravity Center -crop 2000x2000+0+0 -quality 100 -format jpg -path ../cropped *.bmp
cd ..
./shuffle.py