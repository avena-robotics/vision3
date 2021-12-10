import rclpy


def call_empty_service_and_get_data(srv_name: str, srv_type):
    """
    This function is responsible for creating ROS2 service client with specified
    name and type of the service. It handles retrieving data after 
    asynchronous service call succeeded or return None if it fails.
    
    :param srv_name: name of service which should be called,
    :param srv_type: one of the standard ROS2 service types or created by user
    :return object with response or None if it fails
    """
    rclpy.init(args=None)
    node = rclpy.create_node(srv_name + '_node')
    cli = node.create_client(srv_type, srv_name)
    print('Waiting for server for 3 seconds...')
    if not cli.wait_for_service(timeout_sec=3.0):
        print(f'Service "{srv_name}" is not available')
        return None
    req = srv_type.Request()
    future = cli.call_async(req)
    ret_val = None
    while rclpy.ok():
        rclpy.spin_once(node)
        if future.done():
            res: srv_type.Response = future.result()
            if res is not None:
                ret_val = res
            else:
                node.get_logger().error('Exception while calling service: %r' % future.exception())
            break
    node.destroy_node()
    rclpy.shutdown()
    return ret_val
