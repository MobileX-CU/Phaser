#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <zephyr/devicetree.h>
#include <zephyr/sys/util.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
// #include <zephyr/bluetooth/bluetooth.h>
// #include <zephyr/bluetooth/gap.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/pm/pm.h>
#include <zephyr/device.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/time_units.h>

/*
Defs
*/
// ADC macros
#if !DT_NODE_EXISTS(DT_PATH(zephyr_user)) || !DT_NODE_HAS_PROP(DT_PATH(zephyr_user), io_channels)
#error "No suitable devicetree overlay specified"
#endif
#define DT_SPEC_AND_COMMA(node_id, prop, idx) ADC_DT_SPEC_GET_BY_IDX(node_id, idx),

// ADC channels
#define SUPER_CAP_IDX 0
#define MOTOR_CAP_1_IDX 1
#define MOTOR_CAP_2_IDX 2
#define COMMS_IDX 3

// Constants for min val read on ADC for capacitor to discharge/recharge
#define SUPER_CAP_THRESHOLD_MV 2000 //190// Need to tune this value
#define MOTOR_CAP_RELEASE_THRESHOLD_MV 3000//310 //need to tune
#define SUPER_CAP_THRESHOLD (SUPER_CAP_THRESHOLD_MV / 2)
#define MOTOR_CAP_RELEASE_THRESHOLD (MOTOR_CAP_RELEASE_THRESHOLD_MV / 6)

// GPIO
#define CAP_SWITCH_PIN 23  // gpio 0
#define MOTOR_1_CAP_PIN 13 // gpio 1
#define MOTOR_2_CAP_PIN 21 // gpio 0

// Motion
#define STRAIGHT 0
#define LEFT 1
#define RIGHT 2
#define STOP 3

// Timing Constraints (in ms)
// ADC_TIME must be greater than the time it takes to read all ADC channels. readADC() is called ADC_TIME ms before robot_step() is called
// MOTOR_SHUTOFF_TIME is the time the motors are left turned on after being triggered. This value must be smaller than ROBOT_STEP_TIME - ADC_TIME,
// 	otherwise the ADC will be reading the motor capacitors while they are still being discharged
// ROBOT_STEP_TIME is the time between robot steps. 
#define ADC_TIME 10 // Time delay to allow ADC reads of all channels
#define MOTOR_SHUTOFF_TIME 50 // Time delay to shutoff motors after being triggered
#define ROBOT_STEP_TIME 150 // Time delay between robot steps

// // BLE
// #define MSEC_TO_BLE_UNITS(TIME_MS) ((TIME_MS) * 1000 / 625)
// #define MIN_BLUETOOTH_INTERVAL MSEC_TO_BLE_UNITS(500)
// #define MAX_BLUETOOTH_INTERVAL MSEC_TO_BLE_UNITS(501)
// #define DEVICE_NAME CONFIG_BT_DEVICE_NAME
// #define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

// Shared error var
int err;


/*
Bluetooth
*/
// static struct bt_le_adv_param *adv_param = BT_LE_ADV_PARAM(BT_LE_ADV_OPT_NONE, MIN_BLUETOOTH_INTERVAL, MAX_BLUETOOTH_INTERVAL, NULL);
// static const struct bt_data ad[] = { BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_NO_BREDR), BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN), };

// int initBT() {
// 	err = bt_enable(NULL);
// 	if (err) return -1;
// 	err = bt_le_adv_start(adv_param, ad, ARRAY_SIZE(ad), NULL, 0);
// 	if (err) return -1;
// 	return 0;
// }


/*
ADC
*/

// Data of ADC io-channels specified in devicetree
static const struct adc_dt_spec adc_channels[] = {
	DT_FOREACH_PROP_ELEM(DT_PATH(zephyr_user), io_channels, DT_SPEC_AND_COMMA)
};
uint32_t count = 0;
uint16_t buf;
struct adc_sequence sequence = {
	.buffer = &buf,
	.buffer_size = sizeof(buf), // buffer size in bytes, not number of samples
	.resolution = 10,
};

// ADC channels
int32_t adc_outputs_mv[ARRAY_SIZE(adc_channels)];

int initADC() {
	for (size_t i = 0U; i < ARRAY_SIZE(adc_channels); i++) {
		if (!device_is_ready(adc_channels[i].dev)) {
			printk("ADC controller device %s not ready\n", adc_channels[i].dev->name);
			return 1;
		}
		err = adc_channel_setup_dt(&adc_channels[i]);
		if (err < 0) {
			printk("Could not setup channel #%d (%d)\n", i, err);
			return 1;
		}
	}
	return 0;
}

void update_comms(int raw);
void readADC(struct k_work *work) {
    // printk("%u, ADC reading[%u]:\n", k_uptime_get(), count++);
	for (size_t i = 0U; i < ARRAY_SIZE(adc_channels); i++) {
		// printk("- %s, channel %d: ", adc_channels[i].dev->name, adc_channels[i].channel_id);
		(void) adc_sequence_init_dt(&adc_channels[i], &sequence);
		err = adc_read(adc_channels[i].dev, &sequence);
		if (err < 0) {
			//printk("Could not read (%d)\n", err);
			continue;
		}
		if (adc_channels[i].channel_cfg.differential) {
			adc_outputs_mv[i] = (int32_t)((int16_t)buf);
		} else {
			adc_outputs_mv[i] = (int32_t)buf;
		}
		// printk("Raw Value: %d, ", adc_outputs_mv[i]);
		adc_raw_to_millivolts(adc_ref_internal(adc_channels[i].dev), adc_channels[i].channel_cfg.gain, adc_channels[i].resolution, &adc_outputs_mv[i]);
		// printk("MV Value: %d\n", adc_outputs_mv[i]);
		// Update comms when a new sample is ready
		if (i == COMMS_IDX) {
			// update_comms(adc_outputs_mv[i]);
		}
	}
}
K_WORK_DEFINE(adc_work, readADC);
void adc_timer_handler(struct k_timer *dummy) {
	k_work_submit(&adc_work);
}
K_TIMER_DEFINE(adc_timer, adc_timer_handler, NULL);


/* 
GPIO pins/state for capacitors 
*/
uint16_t cap_switch_state = 0; // 0: to supercap, 1: to motor cap
const struct device *gpio0;
const struct device *gpio1;

int initCapGPIO() {
	gpio0 = device_get_binding(DEVICE_DT_NAME(DT_NODELABEL(gpio0)));
	gpio1 = device_get_binding(DEVICE_DT_NAME(DT_NODELABEL(gpio1)));
	int cap_switch_pin_err;
	int motor_1_cap_pin_err;
	int motor_2_cap_pin_err;
	cap_switch_pin_err = gpio_pin_configure(gpio0, CAP_SWITCH_PIN, GPIO_OUTPUT | GPIO_OUTPUT_INIT_LOW);
	motor_1_cap_pin_err = gpio_pin_configure(gpio1, MOTOR_1_CAP_PIN, GPIO_OUTPUT | GPIO_OUTPUT_INIT_LOW);
	motor_2_cap_pin_err = gpio_pin_configure(gpio0, MOTOR_2_CAP_PIN, GPIO_OUTPUT | GPIO_OUTPUT_INIT_LOW);
	if (cap_switch_pin_err < 0 || motor_1_cap_pin_err < 0 || motor_2_cap_pin_err < 0) return -1;
	return 0;
}


/*
Motor shutoff
*/
void motorShutoff(struct k_work *work) {
	gpio_pin_set(gpio1, MOTOR_1_CAP_PIN, 0);
	gpio_pin_set(gpio0, MOTOR_2_CAP_PIN, 0);
}
K_WORK_DEFINE(motor_shutoff_work, motorShutoff);
void motor_shutoff_timer_handler(struct k_timer *dummy) {
	k_work_submit(&motor_shutoff_work);
}
K_TIMER_DEFINE(motor_shutoff_timer, motor_shutoff_timer_handler, NULL);


/*
Motion control
*/
uint16_t motion_state = STRAIGHT;

void robot_step(struct k_work *work);
K_WORK_DEFINE(robot_step_work, robot_step);
void robot_step_timer_handler(struct k_timer *dummy) {
	k_work_submit(&robot_step_work);
}
K_TIMER_DEFINE(robot_step_timer, robot_step_timer_handler, NULL);
void robot_step(struct k_work *work) {
	if (adc_outputs_mv[SUPER_CAP_IDX] <= SUPER_CAP_THRESHOLD) {
		gpio_pin_set(gpio0, CAP_SWITCH_PIN, 0);
		// printk("Switching to supercap\n");
	}
	else {
		gpio_pin_set(gpio0, CAP_SWITCH_PIN, 1);
		// printk("Switching to motor caps, %i\n", motion_state);
		if(adc_outputs_mv[MOTOR_CAP_1_IDX] >= MOTOR_CAP_RELEASE_THRESHOLD && adc_outputs_mv[MOTOR_CAP_2_IDX] >= MOTOR_CAP_RELEASE_THRESHOLD) {
			switch (motion_state) {
				case STRAIGHT:
					gpio_pin_set(gpio1, MOTOR_1_CAP_PIN, 1);
					gpio_pin_set(gpio0, MOTOR_2_CAP_PIN, 1);
					break;
				case LEFT:
					gpio_pin_set(gpio0, MOTOR_2_CAP_PIN, 1);
					break;
				case RIGHT:
					gpio_pin_set(gpio1, MOTOR_1_CAP_PIN, 1);
					break;
				case STOP:
				default:
					break;
			}
			// printk("Motors triggered\n");
			k_timer_start(&motor_shutoff_timer, K_MSEC(MOTOR_SHUTOFF_TIME), K_NO_WAIT);
		}
	}
	k_timer_start(&robot_step_timer, K_MSEC(ROBOT_STEP_TIME), K_NO_WAIT);
	// printk("Robot Step\n");
	return;
}


/*
Comms
*/
#define N 32 // Number of samples in ADC buffer (larger ==> smoother ==> slower)
#define P2P_MIN 10 // Minimum peak-to-peak range [mV]
#define P2P_MAX 120 // Maximum peak-to-peak range [mV]
#define F_LOW 12.5 // Lower frequency bound [Hz]
#define F_HIGH 25 // Upper frequency bound [Hz]
#define DF 5 // Frequency tolerance [Hz]

// Ring buffer of raw ADC samples, used to compute buffer stats for demodulation
int comms_raw[N] = {0};
int n = 0;

// Ring buffer stats
typedef struct buffer_stats {
	int avg;
	int max;
	int min;
	int range;
} buffer_stats;
struct buffer_stats comms_raw_stats;

// Update ring buffer stats, e.g., update_buffer_stats(comms_raw, N, &comms_raw_stats);
void update_buffer_stats(int *buffer, int len, buffer_stats *stats) {
	int max = -100000;
	int min = 100000;
	for (int j = 0; j < len; j++) {
		stats->avg += buffer[j];
		if (buffer[j] <= min) { min = buffer[j]; }
		if (buffer[j] >= max) { max = buffer[j]; }
	}
	stats->min = min;
	stats->max = max;
	stats->avg /= len;
	stats->range = NRFX_ABS(stats->max - stats->min);
}

// Initialization
int initComms() {
	comms_raw_stats.max = 0;
	comms_raw_stats.min = 0;
	comms_raw_stats.avg = 0;
	comms_raw_stats.range = 0;
	return 0;
}

// Rising/falling edge transition times
int64_t t_rise = 0;
int64_t t_rise_prev = 0;
int64_t t_fall = 0;
int64_t t_fall_prev = 0;

// Edge states
#define RISING 1
#define NONE 0
#define FALLING -1
int edge = NONE;

// Logic states
#define LOW 0
#define HIGH 1
int state = LOW;

// Period vars
bool T_too_big = false;
bool T_too_small = false;
int T = 0;

// Valid vars
bool valid_p2p = false;
bool valid_T = false;

void update_comms(int raw) {
	
	// Update samples buffer and compute stats
	comms_raw[n % N] = raw;
	n += 1;
	update_buffer_stats(comms_raw, N, &comms_raw_stats);

	// First check if we have a valid signal strength
	valid_p2p = (comms_raw_stats.range > P2P_MIN) && (comms_raw_stats.range < P2P_MAX);
	
	// State machine to determine edge type
	if (valid_p2p) {
		if (state == LOW) {
			if (raw >= comms_raw_stats.avg) {
				// LOW -> HIGH (rising edge)
				edge = RISING;
				state = HIGH;
				t_rise = k_uptime_get();
			} else {
				// LOW -> LOW (no edge)
				edge = NONE;
			}
		} else if (state == HIGH) {
			if (raw >= comms_raw_stats.avg) {
				// HIGH -> HIGH (no edge)
				edge = NONE;
			} else {
				// HIGH -> LOW (falling edge)
				edge = FALLING;
				state = LOW;
				t_fall = k_uptime_get();
			}
		}
	}

	// Reset signal flags
	T_too_big = false;
	T_too_small = false;

	// Create watchdog to check if it's been too long since a transition
	// We want to do this *before* we calculate the new period, in case there was a transition
	// that reset T_too_big to false, which would override the T check after calculating the actual period 
	if ((state == HIGH) && (k_uptime_get() - t_rise > 1000/(F_LOW - DF))) {
		T_too_big = true;
	} else if ((state == LOW) && (k_uptime_get() - t_fall > 1000/(F_LOW - DF))) {
		T_too_big = true;
	}

	// Calculate the period between subsequent rising edges
	if (edge == RISING) {
		T = k_uptime_get() - t_rise_prev;
		t_rise_prev = t_rise;
	}

	// Calculate if period is within bounds
	if (T > 1000/(F_LOW - DF)) {
		T_too_big = true;
	} else if (T < 1000/(F_HIGH + DF)) {
		T_too_small = true;
	}

	// Determine if the signal is valid based on the period and SNR
	valid_T = !T_too_big & !T_too_small;
	
	// Calculate motor state based on smallest distance to high/low frequencies
	motion_state = STRAIGHT;
	if (valid_T && valid_p2p) {
		int dlow = NRFX_ABS(T-1000/F_LOW);
		int dhigh = NRFX_ABS(T-1000/F_HIGH);
		if (dlow < dhigh) {
			motion_state = LEFT;
		} else {
			motion_state = RIGHT;
		}
		// For BER testing:
		// printk("%i,%i\n", motion_state, comms_raw_stats.range);
	}

	// For distance/range/ambient light testing:
	// printk(">raw:%d\n", raw);
	// printk(">range:%d\n", comms_raw_stats.range);

	// For power testing
	// printk(">running:%i\n", 1);

	// Print 
	// printk(">motion_state:%i\n", motion_state);
	// printk(">valid_p2p:%i\n", valid_p2p);
	// printk(">range:%d\n", comms_raw_stats.range);
	// printk(">T_too_small:%i\n", T_too_small);
	// printk(">T_too_big:%i\n", T_too_big);
	// printk(">valid_T:%i\n", valid_T);
	// printk(">raw:%d\n", raw);
	// printk(">avg:%d\n", comms_raw_stats.avg);
	// printk(">period:%i\n", T);
	// printk(">state:%i\n", state);
}


/*
Main
*/
int main(void) {

	// Initialization
	if (initCapGPIO() != 0) return -1;
	// printk("initCapGPIO() success\n");
	if (initADC() != 0) return -1;
	// printk("initADC() success\n");
	// if (initBT() != 0) return -1;
	// printk("initBT() success\n");
	if (initComms() != 0) return -1;
	// printk("initComms() success\n");

	// Setup timers
	k_timer_start(&adc_timer, K_MSEC(ADC_TIME), K_MSEC(ADC_TIME));
	// printk("adc_timer setup success\n");
	k_timer_start(&robot_step_timer, K_MSEC(ROBOT_STEP_TIME), K_NO_WAIT);
	// printk("robot_step_timer setup success\n");	

	// Loop
	for (;;) {
		k_cpu_idle();
	}
	return 0;
}
