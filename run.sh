# embeddings
curl -O http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-original.EMBEDDING_SIZE=25.txt.gz
gzip -d embeddings-original.EMBEDDING_SIZE=25.txt

# Eigen
curl -L http://bitbucket.org/eigen/eigen/get/3.2.4.tar.gz -o eigen.tar.gz
tar -xzvf eigen.tar.gz --strip-components=1 eigen-eigen-10219c95fe65/Eigen

# compile & run
cmake .
make
./drnt --data agent.txt --lr 0.001
