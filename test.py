import matlab.engine
eng = matlab.engine.start_matlab()
print(eng.sqrt(49.0))
eng.quit()