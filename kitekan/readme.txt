==================================================================================
This dir is for kitekan program to estimate emotion

toriaezu, watch the disc for Ubuntu16.
toriaezu command: `python3 get_factor.py`

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
  - @@@@ super important notes!! @@@@
    - imu data {ax,ay,az} ; ylim(-15,15)
    - 

- basis_function.py

  - is basis functions

- main.py
  - fit_weight(func_type,arg=0,step=0,mode="dummy")
    - input
      - functype represents what the weights is used for.
        from MENTAL, EMOTION, SIGNAL for f(), g(), and h() respectively. 
      - step is step.
      - arg is a set of input and output of the target function.
        - if func_type == MENTAL, arg = factor,mental
        - if func_type == EMOTION, arg = factor,mental,emotion
        - if func_type == SIGNAL, arg = factor,mental,signal

  - method:
    1. i=0で、M_0を暫定で決めておく
    2. i++
    3. i=kまで溜まったら
      3.1 w_fを暫定で決めて、漸化式dM=f(F, M)を使って、暫定M_{i-k}～M_iまで決まる
      3.2 w_h, w_gを暫定で決める
        3.2.1 S=h(E,F,M)を使って、暫定E_h=E_{i-k}～E_iが決まる
            [note] SとFは本当の値を入れる。Mは暫定の値を入れる
        3.2.2 E=g(F,M)を使って、暫定E_g=E_{i-k}～E_iが決まる
      3.3 E_hとE_gの誤差 err(E_h,E_g) を計算する
      3.4 上記3.a~3.cを繰り返して勾配法でw_f,w_h,w_g,Eを探索する
    4. iが10増えたら、上記３を繰り返す、かな。

==================================================================================
