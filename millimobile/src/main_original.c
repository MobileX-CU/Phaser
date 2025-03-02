/*
 * Copyright (c) 2023 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/pm/pm.h>
#include <zephyr/device.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/time_units.h>


//#include <pm/pm.h>  
// #include <device.h> 
void robot_step(struct k_work *work);


//////////////////////// BLE //////////////////////////
#define MSEC_TO_BLE_UNITS(TIME_MS) ((TIME_MS) * 1000 / 625)

#define MIN_BLUETOOTH_INTERVAL MSEC_TO_BLE_UNITS(500)
#define MAX_BLUETOOTH_INTERVAL MSEC_TO_BLE_UNITS(501)

static struct bt_le_adv_param *adv_param =
    BT_LE_ADV_PARAM(BT_LE_ADV_OPT_NONE,
                    MIN_BLUETOOTH_INTERVAL,
                    MAX_BLUETOOTH_INTERVAL,
                    NULL);

#define DEVICE_NAME CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_NO_BREDR),
    BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};

// #define CUSTOM_ADV_TYPE 0xFF 
// #define COMPANY_ID_CODE 0x0059

// typedef struct the_adc_data {
//         uint16_t company_id;
//         int32_t adc_value;
// } the_adc_data_type;

// static the_adc_data_type the_adc_data = {COMPANY_ID_CODE, 0x00};

// static const struct bt_data ad[] = {
//         BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_NO_BREDR),
//         BT_DATA(CUSTOM_ADV_TYPE, (const uint8_t *)&the_adc_data, sizeof(the_adc_data)),
// };

// static struct bt_data sd[1];

// void update_advertising_data(int32_t new_adc_value) {
//         the_adc_data.adc_value = new_adc_value;

//         int err = bt_le_adv_start(adv_param, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
//         if (err) {
//                 printk("Advertising failed to start (Error %d)\n", err);
//         }
// }

// static const struct bt_data sd[] = {
// 	BT_DATA(BT_DATA_URI, url_data, sizeof(url_data)),
// };

//////////////////////// BLE //////////////////////////

//TIMING (all ms)
#define ADC_TIME 10 // Time delay to allow ADC reads of all channels
#define MOTOR_SHUTOFF_TIME 50 // Time delay to shutoff motors after being triggered
#define ROBOT_STEP_TIME 150 // Time delay between robot steps

/*Timing Constraints
ADC_TIME must be greater than the time it takes to read all ADC channels. readADC() is called ADC_TIME ms before robot_step() is called
MOTOR_SHUTOFF_TIME is the time the motors are left turned on after being triggered. This value must be smaller than ROBOT_STEP_TIME - ADC_TIME, 
        otherwise the ADC will be reading the motor capacitors while they are still being discharged
ROBOT_STEP_TIME is the time between robot steps. 

*/


//////////////////////// ADC //////////////////////////


// Constants for min val read on ADC for capacitor to discharge/recharge
#define SUPER_CAP_THRESHOLD_MV 2000 //190// Need to tune this value
#define MOTOR_CAP_RELEASE_THRESHOLD_MV 3000//310 //need to tune
#define SUPER_CAP_THRESHOLD (SUPER_CAP_THRESHOLD_MV / 2)
#define MOTOR_CAP_RELEASE_THRESHOLD (MOTOR_CAP_RELEASE_THRESHOLD_MV / 6)

// Define pins/state for capacitors
uint16_t cap_switch_state = 0; // 0: to supercap, 1: to motor cap
const struct device *gpio0;
const struct device *gpio1;
#define CAP_SWITCH_PIN 23  // gpio 0
#define MOTOR_1_CAP_PIN 13 // gpio 1
#define MOTOR_2_CAP_PIN 21 // gpio 0

// error variable
int err;

// ADC defs
#if !DT_NODE_EXISTS(DT_PATH(zephyr_user)) || \
	!DT_NODE_HAS_PROP(DT_PATH(zephyr_user), io_channels)
#error "No suitable devicetree overlay specified"
#endif

#define DT_SPEC_AND_COMMA(node_id, prop, idx) \
	ADC_DT_SPEC_GET_BY_IDX(node_id, idx),

/* Data of ADC io-channels specified in devicetree. */
static const struct adc_dt_spec adc_channels[] = {
	DT_FOREACH_PROP_ELEM(DT_PATH(zephyr_user), io_channels,
			     DT_SPEC_AND_COMMA)
};
uint32_t count = 0;
	uint16_t buf;
	struct adc_sequence sequence = {
		.buffer = &buf,
		/* buffer size in bytes, not number of samples */
		.buffer_size = sizeof(buf),
                .resolution = 10,
	};

// ADC MV Buffer
int32_t adc_outputs_mv[ARRAY_SIZE(adc_channels)];
#define SUPER_CAP_IDX 0
#define MOTOR_CAP_1_IDX 1
#define MOTOR_CAP_2_IDX 2

K_WORK_DEFINE(robot_step_work, robot_step);

void robot_step_timer_handler(struct k_timer *dummy)
{
        k_work_submit(&robot_step_work);
}
K_TIMER_DEFINE(robot_step_timer, robot_step_timer_handler, NULL);


int initCapGPIO()
{
        gpio0 = device_get_binding(DEVICE_DT_NAME(DT_NODELABEL(gpio0)));
        gpio1 = device_get_binding(DEVICE_DT_NAME(DT_NODELABEL(gpio1)));
        int cap_switch_pin_err;
        int motor_1_cap_pin_err;
        int motor_2_cap_pin_err;

        cap_switch_pin_err = gpio_pin_configure(gpio0, CAP_SWITCH_PIN, GPIO_OUTPUT | GPIO_OUTPUT_INIT_LOW);
        motor_1_cap_pin_err = gpio_pin_configure(gpio1, MOTOR_1_CAP_PIN, GPIO_OUTPUT | GPIO_OUTPUT_INIT_LOW);
        motor_2_cap_pin_err = gpio_pin_configure(gpio0, MOTOR_2_CAP_PIN, GPIO_OUTPUT | GPIO_OUTPUT_INIT_LOW);
        if (cap_switch_pin_err < 0 || motor_1_cap_pin_err < 0 || motor_2_cap_pin_err < 0) return -1;

       // printk("GPIO initialized\n");

        return 0;
}

int initADC()
{
       /* Configure channels individually prior to sampling. */
	for (size_t i = 0U; i < ARRAY_SIZE(adc_channels); i++) {
		if (!device_is_ready(adc_channels[i].dev)) {
			//printk("ADC controller device %s not ready\n", adc_channels[i].dev->name);
			return 1;
		}

		err = adc_channel_setup_dt(&adc_channels[i]);

		if (err < 0) {
			//printk("Could not setup channel #%d (%d)\n", i, err);
			return 1;
		}
	}
        return 0;
}

void readADC(struct k_work *work) //reads all ADC channels and stores mV outputs in adc_outputs_mv array
{
       //printk("ADC reading[%u]:\n", count++);
        for (size_t i = 0U; i < ARRAY_SIZE(adc_channels); i++) {

                //printk("- %s, channel %d: ",
                //        adc_channels[i].dev->name,
                //        adc_channels[i].channel_id);
                

                (void)adc_sequence_init_dt(&adc_channels[i], &sequence);

                err = adc_read(adc_channels[i].dev, &sequence);
                if (err < 0) {
                        //printk("Could not read (%d)\n", err);
                        continue;
                }

                /*
                        * If using differential mode, the 16 bit value
                        * in the ADC sample buffer should be a signed 2's
                        * complement value.
                        */
                if (adc_channels[i].channel_cfg.differential) {
                        adc_outputs_mv[i] = (int32_t)((int16_t)buf);
                } else {
                        adc_outputs_mv[i] = (int32_t)buf;
                }
               // printk("Raw Value: %d, ", adc_outputs_mv[i]);
                //adc_outputs_mv[i] = adc_raw_to_millivolts_dt(&adc_channels[i], &val);
                adc_raw_to_millivolts(adc_ref_internal(adc_channels[i].dev), adc_channels[i].channel_cfg.gain, adc_channels[i].resolution, &adc_outputs_mv[i]);
                //adc_outputs_mv[i] = (int32_t)buf;
               // printk("MV Value: %d\n", adc_outputs_mv[i]);
        }
        // update_advertising_data(adc_outputs_mv[MOTOR_CAP_1_IDX]);
}
K_WORK_DEFINE(adc_work, readADC);

void adc_timer_handler(struct k_timer *dummy)
{
        k_work_submit(&adc_work);
}
K_TIMER_DEFINE(adc_timer, adc_timer_handler, NULL);

void motorShutoff(struct k_work *work)
{
        gpio_pin_set(gpio1, MOTOR_1_CAP_PIN, 0);
        gpio_pin_set(gpio0, MOTOR_2_CAP_PIN, 0);
}

K_WORK_DEFINE(motor_shutoff_work, motorShutoff);

void motor_shutoff_timer_handler(struct k_timer *dummy)
{
        k_work_submit(&motor_shutoff_work);
}
K_TIMER_DEFINE(motor_shutoff_timer, motor_shutoff_timer_handler, NULL);


void robot_step(struct k_work *work)
{       
        if(adc_outputs_mv[SUPER_CAP_IDX] <= SUPER_CAP_THRESHOLD){
                gpio_pin_set(gpio0, CAP_SWITCH_PIN, 0);
                //printk("Switching to supercap\n");
        }
        else
        {
                gpio_pin_set(gpio0, CAP_SWITCH_PIN, 1);
                //printk("Switching to motor caps\n");
                if(adc_outputs_mv[MOTOR_CAP_1_IDX] >= MOTOR_CAP_RELEASE_THRESHOLD && adc_outputs_mv[MOTOR_CAP_2_IDX] >= MOTOR_CAP_RELEASE_THRESHOLD){
                       // printk("motors triggered\n");
                        gpio_pin_set(gpio1, MOTOR_1_CAP_PIN, 1);
                        gpio_pin_set(gpio0, MOTOR_2_CAP_PIN, 1);
                        k_timer_start(&motor_shutoff_timer, K_MSEC(MOTOR_SHUTOFF_TIME), K_NO_WAIT);
                }
        }
        k_timer_start(&adc_timer, K_MSEC(ROBOT_STEP_TIME - ADC_TIME), K_NO_WAIT);
        k_timer_start(&robot_step_timer, K_MSEC(ROBOT_STEP_TIME), K_NO_WAIT);
        //printk("Robot Step\n");
        return;
}
void main(void)
{
	if (initCapGPIO() != 0) return;

	if (initADC() != 0) return;

	k_timer_start(&adc_timer, K_MSEC(ADC_TIME), K_NO_WAIT);
        k_timer_start(&robot_step_timer, K_MSEC(ADC_TIME*2), K_NO_WAIT);

	err = bt_enable(NULL);
	if (err) return;

        err = bt_le_adv_start(adv_param, ad, ARRAY_SIZE(ad), NULL, 0);
        // err = bt_le_adv_start(adv_param, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	if (err) return;

	for (;;) {
		k_cpu_idle();
	}
}
