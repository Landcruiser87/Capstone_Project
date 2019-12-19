import sys, os
from collections import OrderedDict
import time
import threading

from lpmslib import LpmsB2
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
    lputils.logd(TAG, "Connecting sensor 1 ")
    if lpmsb1.connect():
        lputils.logd(TAG, "Connected")
        get_config_register()

    lputils.logd(TAG, "Connecting sensor 2 ")
    if lpmsb2.connect():
        lputils.logd(TAG, "Connected")
        get_config_register()

def disconnect_sensor():
    lputils.logd(TAG, "Disconnecting sensor 1")
    lpmsb1.disconnect()
    lputils.logd(TAG, "Disconnected")

    lputils.logd(TAG, "Disconnecting sensor 2")
    lpmsb2.disconnect()
    lputils.logd(TAG, "Disconnected")

def get_config_register():
    lputils.logd(TAG, "Sensor 1 Config")
    config_reg = lpmsb1.get_config_register()
    print(config_reg)

    lputils.logd(TAG, "Sensor 2 Config")
    config_reg = lpmsb2.get_config_register()
    print(config_reg)

def stream_freq_menu():
    print_menu(stream_freq_menu)
    choice = input(" >>  ")
    exec_menu(stream_freq_menu, choice)
    get_config_register()
    
def set_stream_freq_5Hz():
    lpmsb1.set_stream_frequency_5Hz()
    lpmsb2.set_stream_frequency_5Hz()

def set_stream_freq_10Hz():
    lpmsb1.set_stream_frequency_10Hz()
    lpmsb2.set_stream_frequency_10Hz()

def set_stream_freq_25Hz():
    lpmsb1.set_stream_frequency_25Hz()
    lpmsb2.set_stream_frequency_25Hz()

def set_stream_freq_50Hz():
    lpmsb1.set_stream_frequency_50Hz()
    lpmsb2.set_stream_frequency_50Hz()

def set_stream_freq_100Hz():
    lpmsb1.set_stream_frequency_100Hz()
    lpmsb2.set_stream_frequency_100Hz()

def set_stream_freq_200Hz():
    lpmsb1.set_stream_frequency_200Hz()
    lpmsb2.set_stream_frequency_200Hz()

def set_stream_freq_400Hz():
    lpmsb1.set_stream_frequency_400Hz()
    lpmsb2.set_stream_frequency_400Hz()

def save_parameters():
    lpmsb1.save_parameters()
    lpmsb2.save_parameters()

def set_command_mode():
    lpmsb1.set_command_mode()
    lpmsb2.set_command_mode()

def set_streaming_mode():
    lpmsb1.set_streaming_mode()
    lpmsb2.set_streaming_mode()

def reset_heading():
    lpmsb1.reset_heading()
    lpmsb2.reset_heading()
    get_config_register()

def sync_sensors():
    lpmsb1.start_sync()
    lpmsb2.start_sync()
    time.sleep(1)
    lpmsb1.stop_sync()
    lpmsb2.stop_sync()



def pretty_print_sensor_data(sensor_data):
    j = 25
    d = '.'
    print("IMU ID:".ljust(j, d), sensor_data[0])
    print("TimeStamp:".ljust(j, d), sensor_data[1])
    print("Frame Counter:".ljust(j, d), sensor_data[2])
    print("Battery Level:".ljust(j, d), sensor_data[3])
    print("Battery Voltage:".ljust(j, d), sensor_data[4])
    print("Temperature:".ljust(j, d), sensor_data[5])
    print("Acc:".ljust(j, d), ['%+.3f' % f for f in sensor_data[6]])
    print("Gyr:".ljust(j, d), ['%+.3f' % f for f in sensor_data[7]])
    print("Mag:".ljust(j, d), ['%+.3f' % f for f in sensor_data[8]])
    print("Quat:".ljust(j, d), ['%+.3f' % f for f in sensor_data[9]])
    print("Euler:".ljust(j, d), ['%+.3f' % f for f in sensor_data[10]])
    print("LinAcc:".ljust(j, d), ['%+.3f' % f for f in sensor_data[11]])


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
        sensor_data1 = lpmsb1.get_latest_sensor_data()
        pretty_print_sensor_data(sensor_data1)
        sensor_data2 = lpmsb2.get_latest_sensor_data()
        pretty_print_sensor_data(sensor_data2)
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
    ('p', print_data),
    ('h', reset_heading),
    ('s', sync_sensors),
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


baudrate = 921600

lpmsb1 = LpmsB2.LpmsB2('COM24', baudrate)
lpmsb2 = LpmsB2.LpmsB2('COM64', baudrate)
quit = False

def main():
    global quit
    while not quit:
        print_main_menu()
        choice = input(" >>  ")
        exec_menu(main_menu, choice)

    disconnect_sensor()
    lputils.logd(TAG, "bye")

if __name__ == "__main__":
    # Launch main menu
    main()
