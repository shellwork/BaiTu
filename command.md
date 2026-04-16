
python battleship_ot2_loop.py calibrate

python battleship_ot2_loop.py --strategy prob --seed 42 --robot_ip 169.254.200.128

python battleship_ot2_loop.py reset --robot_ip 169.254.200.128

python battleship_ot2_loop.py --strategy prob --seed 42 --robot_ip 169.254.200.128 --geometry_path calibration.json

python calibrate_geometry.py capture
python calibrate_geometry.py annotate plate_photo_20260416_135456.jpg

skip setup
python battleship_ot2_loop.py --strategy prob --seed 42 --robot_ip 169.254.200.128 --geometry_path calibration.json --skip_setup