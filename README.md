# `chiron_utils`

This repository serves multiple purposes:

- To implement various bots to advise about or play _Diplomacy_
- To provide a library containing various utilities to help implementing _Diplomacy_ bots
- To run games in a more automatic manner than manual invocation

## Installation

Use the provided Makefile to install this project by running the following from the project root directory (the same directory as this README). Ensure the `python` in `PATH` is 3.11 before running this command:

```shell
make install
```

Newer Python versions might work but have not yet been tested.

If the installation process fails, is interrupted, or for any reason needs to be restarted, run `git clean -xdf` to reset the repository's state.

## Usage

To build the default bot container, run the following command:

```shell
make build
```

Once built, you will need to manually handle distributing the generated OCI image.

To use a bot, run the following command:

```shell
docker run --platform=linux/amd64 --rm ghcr.io/allan-dip/chiron-utils [ARGUMENTS]
```

To run a complete game, run the following command:

```shell
python -m chiron_utils.scripts.run_game [ARGUMENTS]
```

Both the bot and game running commands support a `--help` argument to list available options.

## Bots

- [`RandomProposerBot`](src/chiron_utils/bots/random_proposer_bot.py) (`RandomProposerAdvisor` and `RandomProposerPlayer`):
  - Orders are randomly selected from the space of valid moves.
  - Messages are proposals to carry out a set of valid moves, which is also randomly selected. One such proposal is sent to each opponent.
  - Due to the random nature of play, a game consisting entirely of `RandomProposerPlayer`s can last for a very long time. I (Alex) have observed multiple games lasting past 1950 without a clear winner.
  - `RandomProposerPlayer` uses very few resources, so it is useful as stand-ins for other players.
- [`LrBot`](src/chiron_utils/bots/lr_bot.py) (`LrAdvisor` and `LrPlayer`):
  - A logistic regression model is used to predict orders for each available unit, given current game state.
  - To build the bot, run `make build-baseline-lr` to generate the OCI image to run with Docker
    - When running the bot outside of a container, download the latest model file from [lr_models - Google Drive](https://drive.google.com/drive/folders/1FuG3qY51wRkR8RgEBVY49-loln06W-Ro). The filename includes the model release date in `YYYYMMDD` format).
    - Edit the `MODEL_PATH` constant in `lr_bot.py` to point to the unzipped model folder.
  - Code for model training can be found at <https://github.com/ALLAN-DIP/baseline-models>
- LLM advisor bots:
  - A family of bots containing the following types:
    - [`FaafAdvisor`](src/chiron_utils/bots/csu_faaf_advisor_bot.py): A large language model using the FAAF model from the CSU team to provide commentary advice given board states, recommended orders for current player and predicted orders of opponents from Cicero.
    - [`LlmAdvisor`](src/chiron_utils/bots/llm_advisor_bot.py): A large language model using Llama-3.1-8B-Instruct to provide commentary advice given board states, message history, and predicted orders from Cicero.
    - [`LlmNewAdvisor`](src/chiron_utils/bots/llm_advisor_new_bot.py): A large language model using Llama-3.1-8B-Instruct to provide commentary advice given board states, recommended orders for the current player, and predicted orders of opponents from Cicero.
  - To set up the bot:
    - When the bot is first run, it will attempt to download the used model from the Hugging Face Hub. Preparing for this requires multiple steps:
      - Create a Hugging Face account
      - Request access on the page for the following model:
        - [Llama3.1-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
      - Once your request has been approved, authenticate on your local machine using a user access token, using the official [User access tokens](https://huggingface.co/docs/hub/security-tokens) documentation as a guide.
    - When using `FaafAdvisor`, download the [FAAF model checkpoint](https://drive.google.com/file/d/15qGrovFkkOAJd42l1yFtfzzhOCytIbC-/view) and unzip it into the directory [`src/chiron_utils/models/`](src/chiron_utils/models/).
    - When using `LlmAdvisor`, one needs to run another advisor to provide `OPPONENT_MOVE` advice to the same power. For example, one can run the [Cicero advisor](https://github.com/ALLAN-DIP/diplomacy_cicero) with the argument `--advice_levels OPPONENT_MOVE`.
    - When using `FaafAdvisor` or `LlmNewAdvisor`, one needs to run another advisor to provide `MOVE|OPPONENT_MOVE` advice to the same power. For example, one can run the [Cicero advisor](https://github.com/ALLAN-DIP/diplomacy_cicero) with the argument `--advice_levels 'MOVE|OPPONENT_MOVE'`.
  - To use the bot, run the following command from the repository root, replacing `[bot_type]` with the bot's name:
    ```shell
    # Set communication stage to 10 minutes (in seconds) to give enough time
    export COMM_STAGE_LENGTH=600
    python -m chiron_utils.scripts.run_bot --host [host_address] --port [port_address] --game_id [game_id] --power [power_name] --bot_type [bot_type]
    ```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for instructions on how to implement new bots (i.e., advisors and players).

This project uses various code quality tooling, all of which is automatically installed with the rest of the development requirements.

All checks can be run with `make check`, and some additional automatic changes can be run with `make fix`.

To test GitHub Actions workflows locally, install [`act`](https://github.com/nektos/act) and run it with `act`.

## License

[MIT](https://choosealicense.com/licenses/mit/)
