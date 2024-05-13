include ../rvx_python_config.mh

RVX_UTIL_HOME=$(abspath ${CURDIR}/../rvx_util)

GIT_BASE=$(abspath ${CURDIR}/..)
DATASET_DIR=${GIT_BASE}/dataset
VERIFY_DIR=${GIT_BASE}/verify
RESULT_DIR=${CURDIR}/result

NUM_EPOCH=10
NUM_REPEAT=5

APP_NAME_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f cifar-10_l5ccfff
NEURON_TYPE_LIST=q6ssf

$(addsuffix .train, ${APP_NAME_LIST}):%.train:
	@mkdir -p ./result/$(*)
	#@${PYTHON3_CMD} ${RVX_UTIL_HOME}/configure_template.py -i ./Makefile.app.template -o ${RESULT_DIR}/$(*)/Makefile -c GIT_BASE=$(GIT_BASE) APP_NAME=$(*)
	@${PYTHON3_CMD} ./nrx_trainer.py -app $(*) -cmd train test -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

$(addsuffix .train_check, ${APP_NAME_LIST}):%.train_check:
	make $(*).train NEURON_TYPE_LIST=q6ssf NUM_EPOCH=3 NUM_REPEAT=1

test:
	make mnist_l2cf.train_check
