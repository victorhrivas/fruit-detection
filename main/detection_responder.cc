/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
 * SPDX-FileCopyrightText: 2019-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detection_responder.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "driver/i2c.h"


extern "C" {
    #include "i2c-lcd.h"  // Include your C header file here
}
#define I2C_SDA_PIN GPIO_NUM_14
#define I2C_SCL_PIN GPIO_NUM_15
#include "esp_main.h"
#include "driver/gpio.h"
#define LED_PIN GPIO_NUM_4
#if DISPLAY_SUPPORT
#include "image_provider.h"
#include "bsp/esp-bsp.h"

// Camera definition is always initialized to match the trained detection model: 96x96 pix
// That is too small for LCD displays, so we extrapolate the image to 192x192 pix
#define IMG_WD (96 * 2)
#define IMG_HT (96 * 2)

static lv_obj_t *camera_canvas = NULL;
static lv_obj_t *person_indicator = NULL;
static lv_obj_t *label = NULL;

char buffer[10];
static void create_gui(void)
{
  bsp_display_start();
  bsp_display_backlight_on(); // Set display brightness to 100%
  bsp_display_lock(0);
  camera_canvas = lv_canvas_create(lv_scr_act());
  assert(camera_canvas);
  lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

  person_indicator = lv_led_create(lv_scr_act());
  assert(person_indicator);
  lv_obj_align(person_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);
  lv_led_set_color(person_indicator, lv_palette_main(LV_PALETTE_GREEN));

  label = lv_label_create(lv_scr_act());
  assert(label);
  lv_label_set_text_static(label, "Person detected");
  lv_obj_align_to(label, person_indicator, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
  bsp_display_unlock();
}
#endif // DISPLAY_SUPPORT
static esp_err_t i2c_master_init(void)
{
    i2c_port_t i2c_master_port = I2C_NUM_0;

i2c_config_t conf;
conf.mode = I2C_MODE_MASTER;
conf.sda_io_num = I2C_SDA_PIN;
conf.scl_io_num = I2C_SCL_PIN;
conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
conf.master.clk_speed = 100000;
    i2c_param_config(i2c_master_port, &conf);

    return i2c_driver_install(i2c_master_port, conf.mode, 0, 0, 0);
}

void RespondToDetection(float* fruit_score, const char* kCategoryLabels[]) {
  // Find the fruit with the highest score.
  i2c_master_init();
    lcd_init();
    lcd_put_cur(1, 0);
    lcd_send_string("probando ");
  float max_score = 0;
  int max_score_index = 0;
  for (int i = 0; i < 4; ++i) {
    if (fruit_score[i] > max_score) {
      max_score = fruit_score[i];
      max_score_index = i;
    }
  }

    


  // Log the detected fruit.
  if (max_score > 0.5) {
    if (max_score_index == 3) {
      MicroPrintf("No fruit detected");
      lcd_put_cur(1, 0);
      lcd_clear();
        lcd_send_string("No fruit detected");
    }
    else {
      MicroPrintf("Detected fruit: %s", kCategoryLabels[max_score_index]);
      lcd_clear();
      lcd_put_cur(1, 0);
      lcd_send_string((char *)kCategoryLabels[max_score_index]);
      lcd_put_cur(0, 0);
      if (max_score_index == 0){
        lcd_send_string("price: $1.990/Kg");
      }
      else if (max_score_index == 1){
        lcd_send_string("price: $2.490/Kg");
      }
      else if (max_score_index == 2){
        lcd_send_string("price: $1.590/Kg");
      }
      lcd_send_string("Weight: 0.3 Kg");
    }
  } else {
    MicroPrintf("No fruit detected");
    lcd_clear();
    lcd_put_cur(1, 0);
    lcd_send_string("No fruit detected");
  }
  MicroPrintf("Apple: %f, Banana: %f, Lemon: %f, Other: %f", fruit_score[0], fruit_score[1], fruit_score[2], fruit_score[3]);
}
