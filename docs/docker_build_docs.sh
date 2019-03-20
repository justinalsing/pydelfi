docker build -t pydelfi_docs -f Dockerfile .
mkdir -p html
docker run --rm -i --net=none -v $(pwd)/html:/data pydelfi_docs /bin/bash -c "cp -r ./* /data/"
