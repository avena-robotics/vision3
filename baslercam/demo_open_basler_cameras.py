from basler_client_functions.open_basler_cameras import open_basler_cameras

return_value = open_basler_cameras()
print('Successfully opened cameras', return_value)
