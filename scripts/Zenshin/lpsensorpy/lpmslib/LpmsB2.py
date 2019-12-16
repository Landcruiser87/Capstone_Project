from __future__ import absolute_import, division, print_function
import time
import serial
import math
import threading
from threading import Lock
import struct
import sys
from datetime import datetime, timedelta
from lpmslib.LpmsConfig import *
from lpmslib.lputils import *
from lpmslib.LpmsConfigurationSettings import LpmsConfigurationSettings
import collections 

#TODO:
# check serial port opened before executing commands
# add wait for ack routine


class LpmsB2(object):

    PACKET_ADDRESS0     = 0
    PACKET_ADDRESS1     = 1
    PACKET_FUNCTION0    = 2
    PACKET_FUNCTION1    = 3
    PACKET_LENGTH0      = 4
    PACKET_LENGTH1      = 5
    PACKET_RAW_DATA     = 6
    PACKET_LRC_CHECK0   = 7
    PACKET_LRC_CHECK1   = 8
    PACKET_END          = 9

    MAX_DATA_QUEUE      = 50
    
    def __init__(self, port, baudrate):
        self.TAG = "LPMSB2"
        self.port = port
        self.baudrate = baudrate
        self.mutex  = Lock()
        self.__init_params()


    def __clear_params(self):
        self.runOnce = True;
        self.verbose = True
        self.is_thread_running = False
        self.is_sensor_connected = False
        self.debug_log_size = 0
        self.debug_log_size_index =0;
        self.sensor_configuration = LpmsConfigurationSettings()

        self.current_length = 0
        self.current_function = 0
        self.current_address = 0
        self.rx_state = self.PACKET_END
        self.in_bytes = []
        self.rx_buffer = []
        self.raw_tx_data = []
        self.rx_index  = 0
        self.lrc_check = 0
        self.imu_id = 0
        self.timestamp = 0
        self.frame_counter = 0
        self.battery_level = 0
        self.battery_voltage = 0
        self.temperature = 0
        self.acc_x = 0
        self.acc_y = 0
        self.acc_z = 0
        self.gyr_x = 0
        self.gyr_y = 0
        self.gyr_z = 0
        self.mag_x = 0
        self.mag_y = 0
        self.mag_z = 0
        self.angular_vel_x = 0
        self.angular_vel_y = 0
        self.angular_vel_z = 0
        self.quat_w = 0
        self.quat_x = 0
        self.quat_y = 0
        self.quat_z = 0
        self.euler_x = 0
        self.euler_y = 0
        self.euler_z = 0
        self.linacc_x = 0
        self.linacc_y = 0
        self.linacc_z = 0
        self.altitude = 0
        self.pressure = 0
        self.humidity = 0

        self.wait_for_ack = False
        self.wait_for_data = False
        self.wait_for_config = False

        self.lost_packet_count = 0
        self.data_queue = collections.deque([])
        self.latest_data = []

    def __init_params(self):
        self.__clear_params()

    def __thread_is_alive(self):
        try:
            return self.thread.isAlive()
        except AttributeError:
            return False

    def __run(self):
        """ Method that runs forever """
        self.is_thread_running = True
        self.lost_packet_count = 0
        while not self.quit:
            self.is_sensor_connected = True
            bytesToRead = self.serial_port.inWaiting()
            if bytesToRead > 0:
                reading = self.serial_port.read(bytesToRead)
                self.__parse(reading)
            else:
                time.sleep(0.001)

        self.serial_port.close()
        self.is_sensor_connected = False
        self.is_thread_running = False


    # TODO: add offset length check
    def __convert_rxbytes_to_int16(self, offset, dataList):
        """
        dataList is a list
        """
        (i,) = struct.unpack("h", bytearray(dataList[offset:offset+2]))
        return i

    def __convert_rxbytes_to_int(self, offset, dataList):
        """
        dataList is a list
        """
        (i,) = struct.unpack("i", bytearray(dataList[offset:offset+4]))
        return i

    def __convert_rxbytes_to_float(self, offset, dataList):

        """
        dataList is a list
        """
        (i,) = struct.unpack("f", bytearray(dataList[offset:offset+4]))
        return i

    def __convert_int16_to_txbytes(self, v):
        """
        return bytesarray
        """
        return struct.pack("h", v)

    def __convert_int_to_txbytes(self, v):
        """
        return bytesarray
        """
        return struct.pack("i", v)
        
    def __print_str_to_hex(self, s):
        print (":".join("{:02x}".format(c) for c in s))

    # Parser
    def __parse_function(self):
        cf = self.current_function
        if cf == LPMS_ACK:
            logd(self.TAG , "Received Ack")
            self.wait_for_ack = False

        elif cf == LPMS_NACK:
            logd(self.TAG , "Received Nack")
            self.wait_for_ack = False

        elif cf == LPMS_GET_CONFIG:
            self.config_register = self.__convert_rxbytes_to_int(0, self.rx_buffer)
            print ("{0:b}".format(self.config_register))
            self.__parse_configuration_register(self.config_register)
            self.wait_for_config = False

        elif cf == LPMS_GET_SENSOR_DATA:
            if self.sensor_configuration.sixteen_bit_data_enable:
                self.__parse_sensor_data(16)
            else:
                self.__parse_sensor_data()
            self.wait_for_data = False

        elif cf == GET_BATTERY_LEVEL:
            self.battery_level = self.__convert_rxbytes_to_float(0, self.rx_buffer)
            self.wait_for_data = False

        elif cf == GET_CHARGING_STATUS:
            self.chargingStatus = self.__convert_rxbytes_to_int(0, self.rx_buffer)
            self.wait_for_data = False

        elif cf == GET_BATTERY_VOLTAGE:
            self.battery_voltage = self.__convert_rxbytes_to_float(0, self.rx_buffer)
            self.wait_for_data = False

        elif cf == GET_FIRMWARE_VERSION:
            vmajor = self.__convert_rxbytes_to_int(8, self.rx_buffer)
            vminor = self.__convert_rxbytes_to_int(4, self.rx_buffer)
            vbuild = self.__convert_rxbytes_to_int(0, self.rx_buffer)
            self.firmwareVersion = str(vmajor) + "." + str(vminor) + "." + str(vbuild)
            self.wait_for_data = False

        elif cf == GET_PING:
            if self.sensor_configuration.timestamp_counter_mode_enable:
                self.timestamp = self.__convert_rxbytes_to_int(0, self.rx_buffer)
            else:
                self.timestamp = self.__convert_rxbytes_to_float(0, self.rx_buffer)
        
        elif cf == GET_TEMPERATURE:
            self.temperature = self.__convert_rxbytes_to_float(0, self.rx_buffer)
            self.wait_for_data = False

        elif cf == GET_DEBUG_LOGGING_STATUS:
            self.debugLoggingStatus = self.__convert_rxbytes_to_int(0, self.rx_buffer)
            logd(self.TAG , "Debug Logging Status: "+str(self.debugLoggingStatus))
            self.wait_for_data = False

        elif cf == GET_DEBUG_LOG_SIZE:
            self.debug_log_size = self.__convert_rxbytes_to_int(0, self.rx_buffer) // 64
            logd(self.TAG , "Debug Logging Size: "+str(self.debug_log_size))
            self.wait_for_data = False

        elif cf == GET_DEBUG_LOG:
            n = int(self.current_length/64)
            for k in range(n):
                i = k*64
                log = str(self.__convert_rxbytes_to_int(i, self.rx_buffer)) + ','
                for x in range(14):
                    i +=4
                    log += str(self.__convert_rxbytes_to_float(i, self.rx_buffer)) + ','
                log +='\n'

                if self.debug_log_size_index == 0:
                    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + "DebugLog-" + self.TAG + ".csv"
                    logd(self.TAG , "Saving to "+ filename)
                    self.fo = open(filename, "w")
                    self.startTime = datetime.now()

                self.fo.write(log)
                
                self.debug_log_size_index += 1
                self.__update_progress(self.debug_log_size_index)

            if self.debug_log_size_index >= self.debug_log_size:
                self.fo.close()
                dt = (datetime.now()-self.startTime).total_seconds()
                print 
                logd(self.TAG , "Debug log download completed")
                print ("Elapsed time:", str(dt))

            self.wait_for_data = False

    def __update_progress(self, progress):
        percent = int(progress*100/self.debug_log_size)
        sys.stdout.write("\rDownloading: %d%%, %d, %d" % (percent, progress, self.debug_log_size))
        sys.stdout.flush()

    def __parse(self, data):
        self.lrcReceived = 0
        for c in data:
            b = ord(c)
            if self.rx_state == self.PACKET_END:
                if (b == 58):
                    self.rx_state = self.PACKET_ADDRESS0

            elif self.rx_state == self.PACKET_ADDRESS0:
                self.in_bytes = []
                self.in_bytes.append(b)
                self.rx_state = self.PACKET_ADDRESS1

            elif self.rx_state == self.PACKET_ADDRESS1:
                self.in_bytes.append(b)
                self.current_address = self.__convert_rxbytes_to_int16(0, self.in_bytes)
                self.imu_id = self.current_address
                self.rx_state = self.PACKET_FUNCTION0

            elif self.rx_state == self.PACKET_FUNCTION0:
                self.in_bytes = []
                self.in_bytes.append(b)
                self.rx_state = self.PACKET_FUNCTION1

            elif self.rx_state == self.PACKET_FUNCTION1:
                self.in_bytes.append(b)
                self.current_function = self.__convert_rxbytes_to_int16(0, self.in_bytes)
                self.rx_state = self.PACKET_LENGTH0

            elif self.rx_state == self.PACKET_LENGTH0:
                self.in_bytes = []
                self.in_bytes.append(b)
                self.rx_state = self.PACKET_LENGTH1

            elif self.rx_state == self.PACKET_LENGTH1:
                self.in_bytes.append(b)
                self.current_length = self.__convert_rxbytes_to_int16(0, self.in_bytes)
                self.rx_state = self.PACKET_RAW_DATA
                self.rx_index = 0
                self.rx_buffer = []
                
            elif self.rx_state == self.PACKET_RAW_DATA:
                if self.rx_index == self.current_length:
                    self.lrc_check = self.current_address + self.current_function + self.current_length
                    #self.lrc_check = self.lrc_check + sum([ord(c) for c in self.rx_buffer])
                    self.lrc_check = self.lrc_check + sum(self.rx_buffer)
                    self.in_bytes = []
                    self.in_bytes.append(b)
                    self.rx_state = self.PACKET_LRC_CHECK1
                else:
                    # add length check
                    self.rx_buffer.append(b)
                    self.rx_index = self.rx_index + 1
            
            elif self.rx_state == self.PACKET_LRC_CHECK1:
                self.in_bytes.append(b)
                self.lrcReceived = self.__convert_rxbytes_to_int16(0, self.in_bytes)
                if self.lrcReceived == self.lrc_check:
                    self.__parse_function()
                self.rx_state = self.PACKET_END
           
            else:
                self.rx_state = self.PACKET_END

    def __parse_sensor_data(self, data_mode=32):
        o = 0
        r2d = 57.2958
        if data_mode == 16:
            self.converter = lambda offset, l: float(self.__convert_rxbytes_to_int16(offset, l)) / 1000.0
            increment = 2
        else:
            self.converter = lambda offset, l: self.__convert_rxbytes_to_float(offset, l)
            increment = 4

        # TODO: Add timestamp counter mode/elapsed mode
        tsTmp = float(self.__convert_rxbytes_to_int(0, self.rx_buffer))
        if (tsTmp - self.timestamp > self.sensor_configuration.ts_counter_interval):
            self.lost_packet_count = self.lost_packet_count + 1

        self.timestamp = tsTmp

        o += 4
        if self.runOnce:
            self.frame_counter = self.timestamp
            self.runOnce = False
        else:
            self.frame_counter += 1
            
        if self.sensor_configuration.gyro_enable:
            self.gyr_x = self.converter(o, self.rx_buffer) * r2d
            o += increment
            self.gyr_y = self.converter(o, self.rx_buffer) * r2d
            o += increment
            self.gyr_z = self.converter(o, self.rx_buffer) * r2d
            o += increment

        if self.sensor_configuration.accelerometer_enable:
            self.acc_x = self.converter(o, self.rx_buffer)
            o += increment
            self.acc_y = self.converter(o, self.rx_buffer)
            o += increment
            self.acc_z = self.converter(o, self.rx_buffer)
            o += increment

        if self.sensor_configuration.magnetometer_enable:
            self.mag_x = self.converter(o, self.rx_buffer)
            o += increment
            self.mag_y = self.converter(o, self.rx_buffer)
            o += increment
            self.mag_z = self.converter(o, self.rx_buffer)
            o += increment

            # 100 Fixed point
            if data_mode == 16:
                self.mag_x *= 10
                self.mag_y *= 10
                self.mag_z *= 10

        if self.sensor_configuration.angular_velocity_enable:
            self.angular_vel_x = self.converter(o, self.rx_buffer) * r2d
            o += increment
            self.angular_vel_y = self.converter(o, self.rx_buffer) * r2d
            o += increment
            self.angular_vel_z = self.converter(o, self.rx_buffer) * r2d
            o += increment

        if self.sensor_configuration.quaternion_enable:
            self.quat_w = self.converter(o, self.rx_buffer)
            o += increment
            self.quat_x = self.converter(o, self.rx_buffer)
            o += increment
            self.quat_y = self.converter(o, self.rx_buffer)
            o += increment
            self.quat_z = self.converter(o, self.rx_buffer)
            o += increment

        if self.sensor_configuration.euler_enable:
            self.euler_x = self.converter(o, self.rx_buffer) * r2d
            o += increment
            self.euler_y = self.converter(o, self.rx_buffer) * r2d
            o += increment
            self.euler_z = self.converter(o, self.rx_buffer) * r2d
            o += increment

        if self.sensor_configuration.linear_acceleration_enable:
            self.linacc_x = self.converter(o, self.rx_buffer)
            o += increment
            self.linacc_y = self.converter(o, self.rx_buffer)
            o += increment
            self.linacc_z = self.converter(o, self.rx_buffer)
            o += increment

        if self.sensor_configuration.pressure_enable:
            self.pressure = self.converter(o, self.rx_buffer)
            o += increment

            # 10 Fixed point
            if data_mode == 16:
                self.pressure *= 100

        if self.sensor_configuration.altitude_enable:
            self.altitude = self.converter(o, self.rx_buffer)
            o += increment

            # 10 Fixed point
            if data_mode == 16:
                self.altitude *= 100

        if self.sensor_configuration.temperature_enable:
            self.temperature = self.converter(o, self.rx_buffer)
            o += increment

            # 100 Fixed point
            if data_mode == 16:
                self.temperature *= 10


        # Add data to queue
        with self.mutex:
            self.latest_data = []
            self.latest_data.append(self.imu_id)
            self.latest_data.append(self.timestamp)
            self.latest_data.append(self.frame_counter)
            self.latest_data.append(self.battery_level)
            self.latest_data.append(self.battery_voltage)
            self.latest_data.append(self.temperature)
            self.latest_data.append([self.acc_x, self.acc_y, self.acc_z])
            self.latest_data.append([self.gyr_x, self.gyr_y, self.gyr_z])
            self.latest_data.append([self.mag_x, self.mag_y, self.mag_z])
            self.latest_data.append([self.quat_w, self.quat_x, self.quat_y, self.quat_z])
            self.latest_data.append([self.euler_x, self.euler_y, self.euler_z])
            self.latest_data.append([self.linacc_x, self.linacc_y, self.linacc_z])
        
            if len(self.data_queue) > self.MAX_DATA_QUEUE:
                self.data_queue.popleft()
            self.data_queue.append(self.latest_data)

    

    # communication
    def __get_config_register(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        if self.verbose: logd(self.TAG, "Get config register")
        time.sleep(.1)
        self.__lpbus_set_none(LPMS_GET_CONFIG)
        self.wait_for_config = True
        self.__wait_for_response()

    def __send_data(self, function, length):
        txlrc_check = 0
        txBuffer = b'\x3a'
        txBuffer += self.__convert_int16_to_txbytes(self.imu_id)
        txBuffer += self.__convert_int16_to_txbytes(function)
        txBuffer += self.__convert_int16_to_txbytes(length)
        if length > 0:
            txBuffer += self.raw_tx_data
        txlrc_check = self.imu_id + function + length
        if length > 0:
            txlrc_check += sum([ord(c) for c in self.raw_tx_data])
            #txlrc_check += sum(self.raw_tx_data)

        txBuffer += self.__convert_int16_to_txbytes(txlrc_check)
        txBuffer += b'\x0d'
        txBuffer += b'\x0a'
        bytesSent = self.serial_port.write(txBuffer)
        
    def __lpbus_set_none(self, command):
        self.__send_data(command, 0)

    def __lpbus_set_int32(self, command, v):
        self.raw_tx_data = self.__convert_int_to_txbytes(v)
        self.__send_data(command, 4)

    def __lpbus_set_data(self, command, length, dataBuffer):
        self.raw_tx_data = dataBuffer
        self.__send_data(command, length)

    def __wait_for_response(self):
        while self.wait_for_ack or self.wait_for_data or self.wait_for_config:
            time.sleep(.1)

    def __parse_configuration_register(self, cr):
        self.sensor_configuration.parse(cr)


    # User command
    def set_tag(self, tag):
        self.TAG = tag

    def connect(self):
        if self.__thread_is_alive():
            loge(self.TAG, "Another connection established")
            return False

        try:
            self.__clear_params()
            self.thread = threading.Thread(target=self.__run, args=())
            self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)        
            self.quit = False
            logd(self.TAG , "Sensor connected")
            #thread.daemon = True                            # Daemonize thread
            self.thread.start()                              # Start the execution
            time.sleep(1)
            self.set_command_mode()                        # Start the execution
            time.sleep(.2)
            self.__get_config_register()                        # Start the execution
            time.sleep(.2)
            self.set_streaming_mode()
            return True
        except serial.SerialException:
            loge(self.TAG, "Could not open port " + self.port)
            loge(self.TAG, "Please try again")

        return False

    def disconnect(self):
        self.quit = True
        if self.__thread_is_alive():
            self.thread.join()
        logd(self.TAG , "sensor disconnected")
        return True

    def is_connected(self):
        return self.is_sensor_connected

    # Configuration and Status
    def get_config_register(self):
        """
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None

        self.__lpbus_set_none(LPMS_GET_CONFIG)
        self.wait_for_data = True
        self.__wait_for_response()
        """
        return self.sensor_configuration

    def get_status_register(self):
        pass


    # Mode switching
    def set_command_mode(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return False

        if self.verbose: logd(self.TAG, "Set command mode")
        self.__lpbus_set_none(LPMS_GOTO_COMMAND_MODE)
        self.wait_for_ack = True
        logd(self.TAG, "Set command mode wait for response")
        self.__wait_for_response()
        logd(self.TAG, "Set command mode wait for response done")

    def set_streaming_mode(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return False
        self.set_command_mode()
        if self.verbose: logd(self.TAG, "Set streaming mode")
        self.__lpbus_set_none(LPMS_GOTO_STREAM_MODE)
        self.wait_for_ack = True
        self.__wait_for_response()

    # Data transmision
    def get_sensor_data(self):
        """
        get sensor data during command Mode
        """
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return False

        if self.verbose: logd(self.TAG, "Get sensor data")
        self.__lpbus_set_none(LPMS_GET_SENSOR_DATA)
        self.wait_for_data = True
        self.__wait_for_response()
        return self.get_latest_sensor_data()

    def get_latest_sensor_data(self):
        with self.mutex:
            return self.latest_data


    def get_stream_data(self):
        """
        get sensor data during stream Mode
        """
        with self.mutex:
            if len(self.data_queue) > 0:
                return self.data_queue.popleft()
            else:
                return []

    def get_data_queue_length(self):
        return len(self.data_queue)

    def clear_data_queue(self):
        with self.mutex:
            self.data_queue.clear()

    def set_transmit_data(self):
        pass

    def set_stream_frequency(self, freq):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode()
        if self.verbose: logd(self.TAG, "Set stream freq: "+str(freq)+"Hz")
        self.__lpbus_set_int32(LPMS_SET_STREAM_FREQ , freq)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.__get_config_register()
        self.set_streaming_mode()

    def set_stream_frequency_5Hz(self):
        self.set_stream_frequency(LPMS_STREAM_FREQ_5HZ)

    def set_stream_frequency_10Hz(self):
        self.set_stream_frequency(LPMS_STREAM_FREQ_10HZ)

    def set_stream_frequency_25Hz(self):
        self.set_stream_frequency(LPMS_STREAM_FREQ_25HZ)

    def set_stream_frequency_50Hz(self):
        self.set_stream_frequency(LPMS_STREAM_FREQ_50HZ)

    def set_stream_frequency_100Hz(self):
        self.set_stream_frequency(LPMS_STREAM_FREQ_100HZ)

    def set_stream_frequency_200Hz(self):
        self.set_stream_frequency(LPMS_STREAM_FREQ_200HZ)

    def set_stream_frequency_400Hz(self):
        self.set_stream_frequency(LPMS_STREAM_FREQ_400HZ)

    def set_16bit_mode(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode()
        if self.verbose: logd(self.TAG, "Set 16 bit data")
        self.__lpbus_set_int32(LPMS_SET_LPBUS_DATA_MODE, LPMS_LPBUS_DATA_MODE_16)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.__get_config_register()
        self.set_streaming_mode()

    def set_32bit_mode(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode()
        if self.verbose: logd(self.TAG, "Set 32 bit data")
        self.__lpbus_set_int32(LPMS_SET_LPBUS_DATA_MODE, LPMS_LPBUS_DATA_MODE_32)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.__get_config_register()
        self.set_streaming_mode()

    # Register value save and reset
    def save_parameters(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode()
        if self.verbose: logd(self.TAG, "Save parameters to sensor")
        self.__lpbus_set_none(LPMS_WRITE_REGISTERS)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.set_streaming_mode()

    def reset_factory(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode()
        if self.verbose: logd(self.TAG, "Reset factory settings")
        self.__lpbus_set_none(LPMS_RESET_FACTORY_VALUE)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.__get_config_register()
        self.set_streaming_mode()

    # Reference setting and offset reset
    def reset_heading(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode()
        if self.verbose: logd(self.TAG, "Reset heading")
        self.__lpbus_set_int32(LPMS_SET_ORIENTATION_OFFSET, LPMS_ORIENTATION_RESET_HEADING)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.__get_config_register()
        self.set_streaming_mode()

    def set_offset(self, offset):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode() 
        if self.verbose: logd(self.TAG, "Set offset: " + str(offset))
        self.__lpbus_set_int32(LPMS_SET_ORIENTATION_OFFSET, offset)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.__get_config_register()
        self.set_streaming_mode()

    def reset_offset(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.set_command_mode() 
        if self.verbose: logd(self.TAG, "Reset offset")
        self.__lpbus_set_none(LPMS_RESET_ORIENTATION_OFFSET)
        self.wait_for_ack = True
        self.__wait_for_response()
        self.__get_config_register()
        self.set_streaming_mode()


    # Sync
    def start_sync(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        if self.verbose: logd(self.TAG, "Start Sync")
        self.__lpbus_set_none(START_SYNC)
        #self.wait_for_ack = True
        #self.__wait_for_response()

    def stop_sync(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        if self.verbose: logd(self.TAG, "Start Sync")
        self.__lpbus_set_none(STOP_SYNC)
        #self.wait_for_ack = True
        #self.__wait_for_response()

        
    # Debug Log
    def start_debug_logging(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.__lpbus_set_none(START_DEBUG_LOGGING)
        #self.wait_for_ack = True
        #self.__wait_for_response()

    def stop_debug_logging(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.__lpbus_set_none(STOP_DEBUG_LOGGING)
        #self.wait_for_ack = True
        #self.__wait_for_response()

    def clear_debug_log(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.__lpbus_set_none(CLEAR_DEBUG_LOG)
        self.wait_for_ack = True
        self.__wait_for_response()

    def full_flash_erase(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.__lpbus_set_none(FULL_FLASH_ERASE)
        self.wait_for_ack = True
        self.__wait_for_response()

    def get_debug_log_status(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.__lpbus_set_none(GET_DEBUG_LOGGING_STATUS)
        self.wait_for_data = True
        self.__wait_for_response()
        return self.debugLoggingStatus 

    def get_debug_log_size(self):
        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None
        self.__lpbus_set_none(GET_DEBUG_LOG_SIZE)
        self.wait_for_data = True
        self.__wait_for_response()
        return self.debug_log_size

    def get_debug_log(self, indexStart, indexStop):

        if not self.is_connected():
            loge(self.TAG, "sensor not connected")
            return None

        if indexStop < 0:
            loge(self.TAG, "Error index")
            return
        self.debug_log_size = indexStop - indexStart 
        self.debug_log_size_index = 0
        data = self.__convert_int_to_txbytes(indexStart)
        data += self.__convert_int_to_txbytes(indexStop)

        x = indexStart
        n = math.ceil( (indexStop-indexStart)/3)
        for i in range(n):
            data = self.__convert_int_to_txbytes(x)
            y = x+2
            if (y > indexStop):
                y = indexStop
            data += self.__convert_int_to_txbytes(y)
            self.__lpbus_set_data(GET_DEBUG_LOG, 8, data)
            self.wait_for_data = True
            self.__wait_for_response()

            x = y+1

    def reset_lost_packet_count(self):
        self.lost_packet_count = 0
        
    def get_lost_packet_count(self):
        return self.lost_packet_count
    