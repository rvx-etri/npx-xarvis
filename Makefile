include ./rvx_init.mh

config:
	${PYTHON3_CMD} ${RVX_UTIL_HOME}/configure_template.py -i ./rvx_setup.sh.template -o ./rvx_setup.sh -c CURDIR=$(CURDIR)

preinstall:
	yum install sqlite-devel xz-devel # recompile python

install:
	#${PYTHON3_CMD} -m pip install pysqlite
	${PYTHON3_CMD} -m pip install sqlite3
	${PYTHON3_CMD} -m pip install tqdm
	${PYTHON3_CMD} -m pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
	${PYTHON3_CMD} -m pip install snntorch
