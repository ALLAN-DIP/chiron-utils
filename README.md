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
docker run --rm ghcr.io/allan-dip/chiron-utils [ARGUMENTS]
```

To run a complete game, run the following command:

```shell
python -m chiron_utils.scripts.run_games [ARGUMENTS]
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

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for instructions on how to implement new bots (i.e., advisors and players).

This project uses various code quality tooling, all of which is automatically installed with the rest of the development requirements.

All checks can be run with `make check`, and some additional automatic changes can be run with `make fix`.

To test GitHub Actions workflows locally, install [`act`](https://github.com/nektos/act) and run it with `act`.

## License

[MIT](https://choosealicense.com/licenses/mit/)
