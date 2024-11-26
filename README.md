# `chiron_utils`

This repository serves multiple purposes:

- To implement various bots to advise about or play _Diplomacy_
- To provide a library containing various utilities to help implementing _Diplomacy_ bots
- To run games in a more automatic manner than manual invocation

## Installation

Install the environment by running the following from the project root directory (the same directory as this README). Ensure the `python` in `PATH` is 3.11 before running this command:

```shell
pip install -r requirements.txt
```

Newer Python versions might work but have not yet been tested.

Now, need to mount the manila drive for Llama models(Introduction:https://docs.jetstream-cloud.org/general/manilaVM/):
1. Create a mount point on your instance,e.g.
```shell
mkdir /Llama-family
```
2. Configuring the instance

Create the file 
```/etc/ceph/ceph.client.llamashare.keyring``` 
and add the accessKey
```shell
[client.llamashare]
    key = AQAqjBlnGwfWNBAA8UiNEYcyK0s5HWYx7Mm7vg==
```
Also, make sure the permissions on the file are rw to the owner only.
```shell
sudo chmod 600 /etc/ceph/ceph.client.llamashare.keyring
```
3. Edit ```/etc/fstab``` to add the following line(remember to change the address to your created address):
```shell
149.165.158.38:6789,149.165.158.22:6789,149.165.158.54:6789,149.165.158.70:6789,149.165.158.86:6789:/volumes/_nogroup/b901a3c7-a891-46f1-9467-4d05b47987a6/92b22cf8-728b-46b6-aacb-4bacead99116 /Llama-family ceph name=llamashare,x-systemd.device-timeout=30,x-systemd.mount-timeout=30,noatime,_netdev,rw 0 2
```
4. Mount the share
```shell
mount -a
```

5. Add following lines into `~/.bashrc`(change the address accordingly):

```shell
export HF_HOME=/Llama-family/huggingface
export HF_TOKEN_PATH=/Llama-family/huggingface/token
export PYTHONPATH=/home/exouser/yanzewan/chiron-utils/src:$PYTHONPATH
```

LlmAdvisor is done, install CiceroAdvisor following below instructions:

Clone the CICERO repo (https://github.com/ALLAN-DIP/diplomacy_cicero) and run the following commands in the ```diplomacy_cicero/``` directory:
```shell
git checkout alex-dev
TAG=alex-dev make build
~/chiron-utils/scripts/run_cicero.sh alex-dev --host diplomacy.alexhedges.dev --use-ssl --game_id ahedges_2024_10_15_17_23_24 --human_powers AUSTRIA --advice_levels 3
```

Obviously, adjust the arguments like ```--game_id``` as needed. When creating the game, I suggest doing it in the UI and manually set the deadline. I've been using 5m.
You can get run_cicero.sh from https://github.com/ALLAN-DIP/chiron-utils/blob/4a11684d1b58d6e527e519d863f189754fd874d6/scripts/run_cicero.sh. The script assumes that https://js2.jetstream-cloud.org/project/shares/534a349e-9d9a-4757-b820-12d2cb30c76c/ is mounted at /media/volume/cicero-base-models.(The mount method is same as above, but need to change the commend accordingly) Change the hardcoded path accordingly if needed.

## Usage
Modify the ```run_bots.sh``` under ```/chiron_utils/src/chiron_utils/scripts``` to launch the game by running:
```shell
source run_bots.sh
```

## Bots

- [`RandomProposerBot`](src/chiron_utils/bots/random_proposer_bot.py) (`RandomProposerAdvisor` and `RandomProposerPlayer`):
  - Orders are randomly selected from the space of valid moves.
  - Messages are proposals to carry out a set of valid moves, which is also randomly selected. One such proposal is sent to each opponent.
  - Due to the random nature of play, a game consisting entirely of `RandomProposerPlayer`s can last for a very long time. I (Alex) have observed multiple games lasting past 1950 without a clear winner.
  - `RandomProposerPlayer` uses very few resources, so it are useful as stand-ins for other players.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for instructions on how to implement new bots (i.e., advisors and players).

This project uses various code quality tooling, all of which is automatically installed with the rest of the development requirements.

All checks can be run with `make check`, and some additional automatic changes can be run with `make fix`.

To test GitHub Actions workflows locally, install [`act`](https://github.com/nektos/act) and run it with `act`.

## License

[MIT](https://choosealicense.com/licenses/mit/)
