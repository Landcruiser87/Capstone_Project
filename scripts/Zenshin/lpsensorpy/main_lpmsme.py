import sys, os
from collections import OrderedDict
import time
import threading
from lpmslib import LpmsME
from lpmslib import lputils

TAG="MAIN"

def print_menu(menu_options):
    print
    for key,item in menu_options.items():
        print('[' + key.rjust(3,' ') + ' ]' + ': ' + item.__name__)

# Print main menu
def print_main_menu():
    print_menu(main_menu)

# Execute menu
def exec_menu(menu, choice):
    global printer_running

    os.system('cls')
    ch = choice.lower()
    if printer_running:
        stop_print_data()
        while printer_running:
            time.sleep(.1)
        os.system('cls')
    else:
        if ch == '':
            lputils.loge(TAG, "Invalid selection, please try again.")
        else:
            try:
                menu[ch]()
            except KeyError:
                lputils.loge(TAG, "Invalid selection, please try again.")
        return

# Exit program
def exit():
    global quit
    quit = True

# Sensor related commands
def connect_sensor():
    lputils.logd(TAG, "Connecting sensor")
    if lpmsSensor.connect():
        lputils.logd(TAG, "Connected")
        get_config_register()

def disconnect_sensor():
    lputils.logd(TAG, "Disconnecting sensor")
    lpmsSensor.disconnect()
    lputils.logd(TAG, "Disconnected")

def get_config_register():
    config_reg = lpmsSensor.get_config_register()
    print config_reg

def stream_freq_menu():
    print_menu(stream_freq_menu)
    choice = raw_input(" >>  ")
    exec_menu(stream_freq_menu, choice)
    get_config_register()
    
def set_stream_freq_5Hz():
    lpmsSensor.set_stream_frequency_5Hz()

def set_stream_freq_10Hz():
    lpmsSensor.set_stream_frequency_10Hz()

def set_stream_freq_25Hz():
    lpmsSensor.set_stream_frequency_25Hz()

def set_stream_freq_50Hz():
    lpmsSensor.set_stream_frequency_50Hz()

def set_stream_freq_100Hz():
    lpmsSensor.set_stream_frequency_100Hz()

def set_stream_freq_200Hz():
    lpmsSensor.set_stream_frequency_200Hz()

def set_stream_freq_400Hz():
    lpmsSensor.set_stream_frequency_400Hz()

def save_parameters():
    lpmsSensor.save_parameters()

def set_command_mode():
    lpmsSensor.set_command_mode()

def set_streaming_mode():
    lpmsSensor.set_streaming_mode()

def set_16bit_mode():
    lpmsSensor.set_16bit_mode()
    get_config_register()

def set_32bit_mode():
    lpmsSensor.set_32bit_mode()
    get_config_register()

def get_stream_data():
    sensor_data = lpmsSensor.get_stream_data()
    pretty_print_sensor_data(sensor_data)

def get_sensor_data():
    sensor_data = lpmsSensor.get_sensor_data()
    pretty_print_sensor_data(sensor_data)

def calibration_magnetometer():
    lpmsSensor.start_mag_calibration()

def pretty_print_sensor_data(sensor_data):
    j = 25
    d = '.'
    print "IMU ID:".ljust(j, d), sensor_data[0]
    print "TimeStamp:".ljust(j, d), sensor_data[1]
    print "Frame Counter:".ljust(j, d), sensor_data[2]
    print "Temperature:".ljust(j, d), sensor_data[3]
    print "Acc:".ljust(j, d), ['%+.3f' % f for f in sensor_data[4]]
    print "Gyr:".ljust(j, d), ['%+.3f' % f for f in sensor_data[5]]
    print "Mag:".ljust(j, d), ['%+.3f' % f for f in sensor_data[6]]
    print "Quat:".ljust(j, d), ['%+.3f' % f for f in sensor_data[7]]
    print "Euler:".ljust(j, d), ['%+.3f' % f for f in sensor_data[8]]
    print "LinAcc:".ljust(j, d), ['%+.3f' % f for f in sensor_data[9]]


printer_running = False
stop_printing = True
def print_data():
    thread = threading.Thread(target=printer, args=())
    global stop_printing
    stop_printing = False
    thread.start()

def stop_print_data():
    global stop_printing
    global thread
    stop_printing = True
    if printer_running and thread.isAlive():
        thread.join()
        

def printer():
    global stop_printing
    global printer_running
    global printer_running
    printer_running = True
    while not stop_printing:
        os.system('cls')
        get_stream_data()
        time.sleep(.05)
    printer_running = False
    #lputils.logd(TAG, "Printer terminated")

thread = threading.Thread(target=printer, args=())

#######################################
# Global settings
#######################################
main_menu = OrderedDict([
    ('c', connect_sensor),
    ('d', disconnect_sensor),
    ('r', get_config_register),
    ('f', stream_freq_menu),
    ('s', save_parameters),
    ('1', set_command_mode),
    ('2', set_streaming_mode),
    ('16', set_16bit_mode),
    ('32', set_32bit_mode),
    ('3', get_stream_data),
    ('4', get_sensor_data),
    ('m', calibration_magnetometer),
    ('p', print_data),
    ('q', exit),
])

stream_freq_menu = OrderedDict([
    ('0', set_stream_freq_5Hz),
    ('1', set_stream_freq_10Hz),
    ('2', set_stream_freq_25Hz),
    ('3', set_stream_freq_50Hz),
    ('4', set_stream_freq_100Hz),
    ('5', set_stream_freq_200Hz),
    ('6', set_stream_freq_400Hz),

])

port = 'COM106'
baudrate = 115200

lpmsSensor = LpmsME.LpmsME(port, baudrate)
quit = False

def main():
    global quit
    while not quit:
        print_main_menu()
        choice = raw_input(" >>  ")
        exec_menu(main_menu, choice)

    disconnect_sensor()
    lputils.logd(TAG, "bye")

if __name__ == "__main__":
    # Launch main menu
    main()
