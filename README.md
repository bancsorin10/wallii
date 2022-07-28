
# wallii

+ neural network to sort wallpapers between good/bad wannabe
+ written in C
+ will try to judge you by your wallpapers


## input data

The input data is supposed to be raw images (.rgb format).

The input data is not here since it takes a lot of space but you can construct
your own set of data.

You can use a script like:
```
#!/bin/bash

urls=$1

curl $urls \
  -o /tmp/walls

walls=$(cat /tmp/walls | rg -o "href=\".*?\""| rg -o "/w/.*")

readarray -d "\"" -t wwalls <<< "$walls"
ww=${wwalls[@]}
readarray -d " " -t www <<< "$ww"
for url in "${www[@]}";
do
    flag=1
    id=${url:4:6}
    new_url=https://w.wallhaven.cc/full/${id::2}/wallhaven-$id.jpg 
    echo $new_url
    curl $new_url\
      -o $id.jpg

    file $id.jpg | grep "JPEG" || flag=2
    file $id.jpg | grep "JPEG" || rm $id.jpg
    if [[ $flag == 2 ]]
    then
        new_url=https://w.wallhaven.cc/full/${id::2}/wallhaven-$id.png
        echo $new_url
        curl $new_url\
          -o $id.png
        file $id.png | grep 'PNG' || rm $id.png
    fi
done
```

Save it and run it with an argument such as
`https://wallhaven.cc/search?categories=111&purity=110&resolutions=1920x1080&sorting=random&order=desc&colors=000000`.
(For a bigger variety of pictures you could login and watch how the main get
request is made and you can copy it as cURL and replace the simple curl in the
script)

I saved my images under `./images` and you can use python to run
`fast_label.py` in order to place labels fast on the images. You will need the
`pygame` module installed. After you launch the program it will open all images
inside `./images`, and you can hit `RIGHT_ARROW` or `LEFT_ARROW` in order to
sort the images. (Use `ranger` or something similar to correct the names in a
bulk since the script will append `_god` or `_bad` to the end of the name and
mess the file extension)

The image size is hardcoded in the C program to 3888 which is 48x27 resolution.
I have saved these images inside `./inputs`. You can convert your wallpapers
with the imagemagick utilities as such:
```
cd images && for file in $(ls -1); do convert -resize 48x27 $file ../inputs/$file.rgb; done
```

After you have the input images you can do:
```
make

./a.out
```
in order to run the program.

## current status

+ no optimization for how the weights are modified
+ no pruning
+ no validation program (this program only covers the training)
+ trained model is not saved yet :p
