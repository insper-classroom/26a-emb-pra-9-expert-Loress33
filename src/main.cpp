#include "FreeRTOS.h"
#include "ei_accelerometer.h"
#include "ei_analogsensor.h"
#include "ei_at_handlers.h"
#include "ei_classifier_porting.h"
#include "ei_device_raspberry_rp2xxx.h"
#include "ei_dht11sensor.h"
#include "ei_inertialsensor.h"
#include "ei_rp2xxx_internal_temperature.h"
#include "ei_run_impulse.h"
#include "ei_ultrasonicsensor.h"
#include "pico/multicore.h"
#include "pico/stdlib.h"
#include "task.h"
#include <stdio.h>
#include <time.h>
#include <string.h>

// imu
#include <hardware/gpio.h>
#include <hardware/i2c.h>
#include <hardware/uart.h>
#include <pico/stdio.h>

// freertos
#include <FreeRTOS.h>
#include <queue.h>
#include <semphr.h>
#include <stdlib.h>
#include <task.h>

// // específico
#include "mpu6050.h"
// edited
// -- Adicione estas 3 linhas em main.cpp --
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "model-parameters/model_metadata.h"

using namespace ei;

extern "C" EI_IMPULSE_ERROR
run_classifier(ei::signal_t *signal, ei_impulse_result_t *result, bool debug);

static bool debug_nn = false;

const int MPU_ADDRESS = 0x68;
const int I2C_SDA_GPIO = 4;
const int I2C_SCL_GPIO = 5;
const int LED_PIN_R = 16;
const int LED_PIN_G = 15;
const int LED_PIN_B = 14;

QueueHandle_t xQueueLabel;

static void mpu6050_init() {
	i2c_init(i2c_default, 400 * 1000);
	gpio_set_function(I2C_SDA_GPIO, GPIO_FUNC_I2C);
	gpio_set_function(I2C_SCL_GPIO, GPIO_FUNC_I2C);
	gpio_pull_up(I2C_SDA_GPIO);
	gpio_pull_up(I2C_SCL_GPIO);

	// Two byte reset. First byte register, second byte data
	// There are a load more options to set up the device in different ways that could be added here
	uint8_t buf[] = { 0x6B, 0x00 };
	i2c_write_blocking(i2c_default, MPU_ADDRESS, buf, 2, false);
}

static void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3], int16_t *temp) {
	uint8_t buffer[14];

	// Read all data sequentially starting from acceleration registers (0x3B)
	// 0x3B-0x40: acceleration (6 bytes)
	// 0x41-0x42: temperature (2 bytes)
	// 0x43-0x48: gyro (6 bytes)
	uint8_t val = 0x3B;
	i2c_write_blocking(i2c_default, MPU_ADDRESS, &val, 1, true);
	i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 14, false);

	// Parse acceleration
	for (int i = 0; i < 3; i++) {
		accel[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);
	}

	// Parse temperature
	*temp = buffer[6] << 8 | buffer[7];

	// Parse gyro
	for (int i = 0; i < 3; i++) {
		gyro[i] = (buffer[8 + i * 2] << 8 | buffer[8 + (i * 2) + 1]);
	}
}

static void gesture_recognize_task(void *p) {
	mpu6050_init();
	int16_t accelerometer[3], gyro[3], temp;
	float max_value = -1.0f;
	char predict[32];
	predict[0] = '\0';

	while (true) {
		printf("\n=== Iniciando coleta de dados ===\n");

		float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

		for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
			mpu6050_read_raw(accelerometer, gyro, &temp);
			buffer[ix + 0] = accelerometer[0];
			buffer[ix + 1] = accelerometer[1];
			buffer[ix + 2] = accelerometer[2];

			vTaskDelay(pdMS_TO_TICKS(10));
		}
		printf("Coleta concluida! Buffer size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);

		// Prepara sinal
		ei::signal_t signal;
		int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
		if (err != 0) {
			printf("ERRO ao criar sinal: %d\n", err);
			vTaskDelay(pdMS_TO_TICKS(1000));
			continue;
		}
		printf("Sinal criado com sucesso!\n");

		// Run the classifier
		ei_impulse_result_t result = { 0 };
		printf("Executando classificador...\n");
		err = run_classifier(&signal, &result, debug_nn);
		if (err != EI_IMPULSE_OK) {
			printf("ERRO ao rodar classificador: %d\n", err);
			vTaskDelay(pdMS_TO_TICKS(1000));
			continue;
		}
		printf("Classificador executado!\n");

		// print the predictions
		// printf("\n========== PREDICOES ==========\n");
		// printf(
		// 	"Tempos - DSP: %d ms, Classificacao: %d ms, Anomalia: %d ms\n",
		// 	result.timing.dsp,
		// 	result.timing.classification,
		// 	result.timing.anomaly);
		// printf("Numero de classes: %d\n", EI_CLASSIFIER_LABEL_COUNT);

		for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
			if (result.classification[ix].value > max_value) {
				max_value = result.classification[ix].value;
				snprintf(predict, sizeof(predict), "%s", result.classification[ix].label);
			}
			// printf(
			// 	"%s: %.5f (%.2f%%)\n",
			// 	result.classification[ix].label,
			// 	result.classification[ix].value,
			// 	result.classification[ix].value * 100.0f);
		}
		
		printf("Result: %s\n", predict);
		if (xQueueSend(xQueueLabel, predict, pdMS_TO_TICKS(100)) != pdPASS) {
			/* failed to send - optionally handle */
		}

#if EI_CLASSIFIER_HAS_ANOMALY == 1
		printf("Anomalia: %.3f\n", result.anomaly);
#endif
		printf("===============================\n");
		vTaskDelay(pdMS_TO_TICKS(2000));
	}
}

void led_task (void *p) {
	char label[32];
	while (1) {
		if (xQueueReceive(xQueueLabel, label, pdMS_TO_TICKS(100))) {
			printf("Received prediction: %s\n", label);
			if (strcmp(label, "idle") == 0) {
				gpio_put(LED_PIN_R, 1);
				gpio_put(LED_PIN_G, 0);
				gpio_put(LED_PIN_B, 0);
			} else if (strcmp(label, "wave") == 0) {
				gpio_put(LED_PIN_R, 0);
				gpio_put(LED_PIN_G, 1);
				gpio_put(LED_PIN_B, 0);
			} else if (strcmp(label, "updown") == 0) {
				gpio_put(LED_PIN_R, 0);
				gpio_put(LED_PIN_G, 0);
				gpio_put(LED_PIN_B, 1);
			}
		}
		vTaskDelay(pdMS_TO_TICKS(50));
	}
}

int main(void) {
	stdio_init_all();
	
	gpio_init(LED_PIN_R);
	gpio_set_dir(LED_PIN_R, GPIO_OUT);
	
	gpio_init(LED_PIN_G);
	gpio_set_dir(LED_PIN_G, GPIO_OUT);
	
	gpio_init(LED_PIN_B);
	gpio_set_dir(LED_PIN_B, GPIO_OUT);
	
	xQueueLabel = xQueueCreate(2, 32);

	xTaskCreate(gesture_recognize_task, "gesture_task 1", 8192, NULL, 1, NULL);
	xTaskCreate(led_task, "led_task 1", 2048, NULL, 1, NULL);
	vTaskStartScheduler();

	while (true)
		;
}
