include ./rvx_init.mh

preinstall:
	yum install sqlite-devel xz-devel # recompile python

install:
	${PYTHON3_CMD} -m pip install pysqlite tqdm snntorch
	${PYTHON3_CMD} -m pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
