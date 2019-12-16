import sys, os
from collections import OrderedDict
import time
import threading

from lpmslib import LpmsB2
from lpmslib import lputils

# Connection settings
port = 'COM64'
baudrate = 921600

lpmsb = LpmsB2.LpmsB2(port, baudrate)
quit = False

# Test settings
test_duration = 30 #seconds
stream_frequency = 200
dt = 1.0/stream_frequency*1.5
n_max = test_duration * stream_frequency


if lpmsb.connect():
    while not lpmsb.is_connected():
        time.sleep(1)
    print("Sensor connected")

    lpmsb.set_stream_frequency(stream_frequency)


    print("Stream frequency: %d" % (stream_frequency))
    print("Test Duration: %d" % (test_duration))
    print("Collecting %d data samples" % (n_max))
    raw_input("Press enter to continue")

    # Init
    n = 0
    last_t = 0
    start_time = time.time()
    lost_packet_count = 0
    lpmsb.lost_packet_count = 0
    lpmsb.clear_data_queue()
    print_counter = 0

    while not quit:
        # Get sensor data
        data_queue_length = lpmsb.get_data_queue_length()
        sensor_data = lpmsb.get_stream_data()
        if not sensor_data:
            continue

        # Extract sensor data
        timestamp = sensor_data[1]*0.0025 # convert to seconds
        accData = sensor_data[6]
        gyroData = sensor_data[7]

        # Lost packet analysis
        if last_t == 0:
            last_t = timestamp

        elif last_t != timestamp:
            n = n + 1
            print_counter = print_counter + 1

            # Only print out data every 10 samples
            if print_counter % 10 == 0:
                print_counter = 0
                print(str(n).ljust(5, " "),
                	"Q: %d" % data_queue_length,
                    "TS: %.3f" %timestamp, 
                    "ACC:", ['%+.3f' % f for f in accData], 
                    "GYRO:", ['%+.3f' % f for f in gyroData])

            if (timestamp - last_t > dt):
                lost_packet_count = lost_packet_count + 1
            last_t = timestamp


        if n >= n_max:
            quit = True
        

    # Summary
    elapsed_time = time.time() - start_time
    print("Elapsed time(s): ", elapsed_time)
    print("Diff time(s): ", elapsed_time - test_duration)
    print("Lost packet: ", lost_packet_count)
    print("Library Lost packet: ", lpmsb.get_lost_packet_count())
    lpmsb.disconnect()
