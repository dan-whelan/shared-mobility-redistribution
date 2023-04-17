# COLORS
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)


TARGET_MAX_CHAR_NUM=20
## Show help
help:
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-$(TARGET_MAX_CHAR_NUM)s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

## Run Random Taxi Generator
random-taxi:
	@python3 ./RLWithTaxiEnv/taxi-random.py

## Run Q-Learning Algo with Taxi Implementation
q-learning-taxi:
	@python3 ./RLWithTaxiEnv/taxi-q-learning.py

## Run SARSA Algo with Taxi implementation
sarsa-taxi:
	@python3 ./RLWithTaxiEnv/taxi-SARSA.py

## Run Expected-SARSA Algo with Taxi implementation
expected-sarsa-taxi:
	@python3 ./RLWithTaxiEnv/taxi-Expected-Sarsa.py

## Run Double Q-Learning Algo with Taxi implementation
double-q-taxi:
	@python3 ./RLWithTaxiEnv/taxi-Double-Q-Learning.py

## Run Q-Learning Algo with Custom Implementation
q-learning-model:
	@python3 ./RLWithOwnModel/model-q-learning.py

## Run SARSA Algo with Custom Implementation
sarsa-model:
	@python3 ./RLWithOwnModel/model-sarsa.py

## Run Expected-SARSA Algo with Custom Implementation
expected-sarsa-model:
	@python3 ./RLWithOwnModel/model-expected-sarsa.py

## Run Double Q-Learning Algo with Custom Implementation
double-q-model:
	@python3 ./RLWithOwnModel/model-double-q-learning.py
