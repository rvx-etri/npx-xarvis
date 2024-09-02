include ../rvx_python_config.mh

RVX_UTIL_HOME=$(abspath ${CURDIR}/../rvx_util)

GIT_BASE=$(abspath ${CURDIR}/..)
DATASET_DIR=${GIT_BASE}/dataset
VERIFY_DIR=${GIT_BASE}/verify
RESULT_DIR=${CURDIR}/result

NUM_EPOCH=10
NUM_REPEAT=5
NUM_SAMPLE=3

train:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd train quantize test -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

riscv: train
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(CFG_FILE) -o ${RESULT_DIR}

reset:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd reset -epoch ${NUM_EPOCH} -repeat ${NUM_REPEAT} -dataset ${DATASET_DIR} -o ${RESULT_DIR}

tv: riscv
	@make riscv CFG_FILE=$(CFG_FILE)
	@${PYTHON3_CMD} ./npx_testvector.py -cfg $(CFG_FILE) -cmd testvector -dataset ${DATASET_DIR} -s ${NUM_SAMPLE} -o ${RESULT_DIR}

APP_CFG_DIR=${CURDIR}/app
APP_CFG_LIST=$(wildcard ${APP_CFG_DIR}/*.cfg)
APP_CFG_NAME_LIST=$(basename $(notdir ${APP_CFG_LIST}))

$(addsuffix .train, ${APP_CFG_NAME_LIST}):%.train:
	@make train CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .riscv, ${APP_CFG_NAME_LIST}):%.riscv:
	@make riscv CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .reset, ${APP_CFG_NAME_LIST}):%.reset:
	@make reset CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .tv, ${APP_CFG_NAME_LIST}):%.tv:
	@make tv CFG_FILE=${APP_CFG_DIR}/$(*).cfg

TEST_APP_CFG_FILE=${APP_CFG_DIR}/mnist_app.cfg

check1:
	@make train CFG_FILE=${TEST_APP_CFG_FILE} NUM_EPOCH=3 NUM_REPEAT=2

check2:
	@make riscv CFG_FILE=${TEST_APP_CFG_FILE} NUM_EPOCH=3 NUM_REPEAT=2

check3:
	@make tv CFG_FILE=${TEST_APP_CFG_FILE} NUM_EPOCH=3 NUM_REPEAT=2 NUM_SAMPLE=${NUM_SAMPLE}

APP_NAME_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f cifar10_l5cccff gtsrb_l5cccff
NEURON_TYPE_LIST=q8ssf

$(addsuffix .gen_cfg, ${APP_NAME_LIST}):%.gen_cfg:
	@${PYTHON3_CMD} ./npx_app_cfg_generator.py -app $(*) -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -o ${APP_CFG_DIR}

GENERATED_CFG_DIR=generated_cfg

$(addsuffix .check_cfg, ${APP_NAME_LIST}):%.check_cfg:
	@make $(*).gen_cfg APP_CFG_DIR=${GENERATED_CFG_DIR}
	@make $(*).riscv APP_CFG_DIR=${GENERATED_CFG_DIR}
