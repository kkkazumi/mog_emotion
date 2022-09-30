This dir is for kitekan program to estimate emotion

===
* assumed factors

  1. points, d{points}
  2. timing of hitting
  3. success and failure
  4. position of mogura
  5. motion of hitting

===

- config.ini
 
  - setting file which has the sample rate of each type of sensors. tabun.

      ```
      [SAMPLE_RATE]
      imu_rate = 1771
      fsr_rate = 27000
      photo_rate = 27000

      [WINDOW_SIZE]
      factor_estimation_period_sec = 0.5
      smoothing_length = 100
      ```

- get_factor.py

  - is to get data and output the state of factors estimated based on the data.

- basis_function.py

  - is basis functions
