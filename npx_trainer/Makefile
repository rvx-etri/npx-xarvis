include ../rvx_python_config.mh

RVX_UTIL_HOME=$(abspath ${CURDIR}/../rvx_util)

GIT_BASE=$(abspath ${CURDIR}/..)
DATASET_DIR=${GIT_BASE}/dataset
VERIFY_DIR=${GIT_BASE}/verify
RESULT_DIR=${CURDIR}/result

NUM_EPOCH=10
NUM_REPEAT=1

APP_NAME_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f cifar10_l5cccff gtsrb_l5cccff
NEURON_TYPE_LIST=q8ssf
CFG=mnist.cfg

$(addsuffix .train, ${APP_NAME_LIST}):%.train:
	@mkdir -p ./result/$(*)
	#@${PYTHON3_CMD} ${RVX_UTIL_HOME}/configure_template.py -i ./Makefile.app.template -o ${RESULT_DIR}/$(*)/Makefile -c GIT_BASE=$(GIT_BASE) APP_NAME=$(*)
	@${PYTHON3_CMD} ./npx_trainer.py -app $(*) -cmd train test -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

train_cfg:
	@mkdir -p ./result/$(*)
	@${PYTHON3_CMD} ./npx_trainer.py -cfg ${CFG} -cmd train_cfg test_cfg -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

$(addsuffix .train_check, ${APP_NAME_LIST}):%.train_check:
	make $(*).train NEURON_TYPE_LIST=q6ssf NUM_EPOCH=3 NUM_REPEAT=1

test:
	make mnist_l2cf.train_check

cfg:
	${PYTHON3_CMD} ./npx_gen_cfg.py -cfg ${CFG} -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

#$(addsuffix .test, mnist_l2cf_q6ssf):%.test:
#	#@${PYTHON3_CMD} ./npx_gen_cfg.py -app $(*) -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -o ${RESULT_DIR}
#	@${PYTHON3_CMD} ./npx_gen_cfg.py -cfg $(*) -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

bin:
	${PYTHON3_CMD} ./npx_save_binary.py -cfg ${CFG} -cmd inference test binary -neuron ${NEURON_TYPE_LIST} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

test_vector:
	${PYTHON3_CMD} ./npx_save_binary.py -cfg ${CFG} -cmd test_vector -neuron ${NEURON_TYPE_LIST} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}