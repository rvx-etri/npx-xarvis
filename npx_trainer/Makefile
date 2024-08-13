include ../rvx_python_config.mh

RVX_UTIL_HOME=$(abspath ${CURDIR}/../rvx_util)

GIT_BASE=$(abspath ${CURDIR}/..)
DATASET_DIR=${GIT_BASE}/dataset
VERIFY_DIR=${GIT_BASE}/verify
RESULT_DIR=${CURDIR}/result

NUM_EPOCH=2
NUM_REPEAT=1

train:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd train quantize test -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

riscv:
	@make train CFG_FILE=$(CFG_FILE)
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(CFG_FILE) -cmd bin -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

reset:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd reset -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

tv:
	@make train CFG_FILE=$(CFG_FILE)
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(CFG_FILE) -cmd test_vector -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

EXAMPLE_APP_CFG_DIR=${CURDIR}/app
APP_CFG_LIST=$(wildcard ${EXAMPLE_APP_CFG_DIR}/*.cfg)
APP_CFG_NAME_LIST=$(basename $(notdir ${APP_CFG_LIST}))

$(addsuffix .train, ${APP_CFG_NAME_LIST}):%.train:
	@make train CFG_FILE=${EXAMPLE_APP_CFG_DIR}/$(*).cfg

$(addsuffix .riscv, ${APP_CFG_NAME_LIST}):%.riscv:
	@make riscv CFG_FILE=${EXAMPLE_APP_CFG_DIR}/$(*).cfg

$(addsuffix .reset, ${APP_CFG_NAME_LIST}):%.reset:
	@make reset CFG_FILE=${EXAMPLE_APP_CFG_DIR}/$(*).cfg

$(addsuffix .tv, ${APP_CFG_NAME_LIST}):%.tv:
	@make tv CFG_FILE=${EXAMPLE_APP_CFG_DIR}/$(*).cfg

test:
	@make train CFG_FILE=${EXAMPLE_APP_CFG_DIR}/mnist_app.cfg

APP_NAME_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f cifar10_l5cccff gtsrb_l5cccff
NEURON_TYPE_LIST=q8ssf

$(addsuffix .exam_cfg, ${APP_NAME_LIST}):%.exam_cfg:
	@${PYTHON3_CMD} ./npx_exam_app.py -app $(*) -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -p ${EXAMPLE_APP_CFG_DIR}

$(addsuffix .exam_train, ${APP_NAME_LIST}):%.exam_train:
	@mkdir -p ./result/$(*)
	@make $(*).exam_cfg
	@make $(*).train

$(addsuffix .exam_riscv, ${APP_NAME_LIST}):%.exam_riscv:
	@mkdir -p ./result/$(*)
	@make $(*).exam_cfg
	@make $(*).riscv
