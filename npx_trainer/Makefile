include ../rvx_python_config.mh

RVX_UTIL_HOME=$(abspath ${CURDIR}/../rvx_util)

GIT_BASE=$(abspath ${CURDIR}/..)
DATASET_DIR=${GIT_BASE}/dataset
VERIFY_DIR=${GIT_BASE}/verify
RESULT_DIR=${CURDIR}/result
NET_CFG_DIR=${CURDIR}/net_cfg

NUM_EPOCH=10
NUM_REPEAT=1

APP_NAME_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f cifar10_l5cccff gtsrb_l5cccff
NEURON_TYPE_LIST=q8ssf
NET_CFG_LIST=$(wildcard ${NET_CFG_DIR}/*.cfg)
NET_CFG_NAME_LIST=$(basename $(notdir ${NET_CFG_LIST}))

TEST=$(notdir ${NET_CFG_LIST})
#$(addsuffix .train, ${APP_NAME_LIST}):%.train:
#	@mkdir -p ./result/$(*)
#	#@${PYTHON3_CMD} ${RVX_UTIL_HOME}/configure_template.py -i ./Makefile.app.template -o ${RESULT_DIR}/$(*)/Makefile -c GIT_BASE=$(GIT_BASE) APP_NAME=$(*)
#	@${PYTHON3_CMD} ./npx_trainer.py -app $(*) -cmd train test -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

tmp:
	@echo $(TEST)

$(addsuffix .train, ${NET_CFG_NAME_LIST}):%.train:
	@mkdir -p ./result/$(*)
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(*).cfg -cmd train test -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${NET_CFG_DIR}

$(addsuffix .bin, ${NET_CFG_NAME_LIST}):%.bin:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(*).cfg -cmd inference sim binary -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${NET_CFG_DIR}
	#@${PYTHON3_CMD} ./npx_trainer.py -cfg $(*).cfg -cmd binary -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${NET_CFG_DIR}

$(addsuffix .tv, ${NET_CFG_NAME_LIST}):%.tv:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(*).cfg -cmd test_vector -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${NET_CFG_DIR}

$(addsuffix .reset, ${NET_CFG_NAME_LIST}):%.reset:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(*).cfg -cmd reset -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${NET_CFG_DIR}

$(addsuffix .train_check, ${APP_NAME_LIST}):%.train_check:
	make $(*).train NEURON_TYPE_LIST=q6ssf NUM_EPOCH=3 NUM_REPEAT=1

test:
	make mnist_l2cf.train_check

$(addsuffix .app_train, ${APP_NAME_LIST}):%.app_train:
	@mkdir -p ./result/$(*)
	@${PYTHON3_CMD} ./npx_trainer.py -app $(*) -cmd app_cfg train test -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${NET_CFG_DIR}

$(addsuffix .app_cfg, ${APP_NAME_LIST}):%.app_cfg:
	@${PYTHON3_CMD} ./npx_trainer.py -app $(*) -cmd app_cfg -neuron ${NEURON_TYPE_LIST} -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${NET_CFG_DIR}
#cfg:
#	${PYTHON3_CMD} ./npx_gen_cfg.py -cfg ${NET_CFG_LIST} -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -o ${RESULT_DIR}