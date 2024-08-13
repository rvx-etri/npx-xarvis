include ../rvx_python_config.mh

RVX_UTIL_HOME=$(abspath ${CURDIR}/../rvx_util)

GIT_BASE=$(abspath ${CURDIR}/..)
DATASET_DIR=${GIT_BASE}/dataset
VERIFY_DIR=${GIT_BASE}/verify
RESULT_DIR=${CURDIR}/result
APP_CFG_DIR=${CURDIR}/app

NUM_EPOCH=10
NUM_REPEAT=5

APP_NAME_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f cifar10_l5cccff gtsrb_l5cccff
NEURON_TYPE_LIST=q8ssf
APP_CFG_LIST=$(wildcard ${APP_CFG_DIR}/*.cfg)
APP_CFG_NAME_LIST=$(basename $(notdir ${APP_CFG_LIST}))

$(addsuffix .train, ${APP_CFG_NAME_LIST}):%.train:
	@mkdir -p ./result/$(*)
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(*).cfg -cmd train quantize test -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${APP_CFG_DIR}

$(addsuffix .riscv, ${APP_CFG_NAME_LIST}):%.riscv:
	@make $(*).train
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(*).cfg -cmd bin -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${APP_CFG_DIR}

$(addsuffix .reset, ${APP_CFG_NAME_LIST}):%.reset:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(*).cfg -cmd reset -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${APP_CFG_DIR}

$(addsuffix .inf, ${APP_CFG_NAME_LIST}):%.inf:
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(*).cfg -cmd inf -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${APP_CFG_DIR}

$(addsuffix .sim, ${APP_CFG_NAME_LIST}):%.sim:
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(*).cfg -cmd sim -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${APP_CFG_DIR}

$(addsuffix .tv, ${APP_CFG_NAME_LIST}):%.tv:
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(*).cfg -cmd test_vector -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR} -p ${APP_CFG_DIR}

#$(addsuffix .train_check, ${APP_NAME_LIST}):%.train_check:
#	make $(*).train NEURON_TYPE_LIST=q6ssf NUM_EPOCH=3 NUM_REPEAT=1

#test:
#	make mnist_l2cf.train_check

$(addsuffix .exam_cfg, ${APP_NAME_LIST}):%.exam_cfg:
	@${PYTHON3_CMD} ./npx_exam_app.py -app $(*) -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -p ${APP_CFG_DIR}

$(addsuffix .exam_train, ${APP_NAME_LIST}):%.exam_train:
	@mkdir -p ./result/$(*)
	@make $(*).exam_cfg
	@make $(*).train

$(addsuffix .exam_riscv, ${APP_NAME_LIST}):%.exam_riscv:
	@mkdir -p ./result/$(*)
	@make $(*).exam_cfg
	@make $(*).riscv
