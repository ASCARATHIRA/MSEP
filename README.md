The project contains a directory "core" where the code is located. So the project home is accessed from the code via the relative path "..".
To reproduce the results follow the following steps:
1. Download senna and unzip it to a home directory "senna" in the project home so that it can be accessed from the code as "../senna".
2. Compile senna from source or create a symlink named "senna" to the executable provided in the "senna" directory.
3. Download the pretrained vectors from the following locations:
	a. Google: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
	b. LexVec: http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki%2bnewscrawl.300d.W.pos.vectors.gz
	c. Glove: http://nlp.stanford.edu/data/glove.6B.zip
4. Unzip the vectors and place them in the project home directory so that they can be accessed via "../path-to-vectors"
5. Modify the paths to the vectors in the code, if necessary
6. Unzip "nw.zip" to a directory "nw" in the project home so that the code can access it via "../nw".
7. Run the code with "python main.py" from the "core" directory.
