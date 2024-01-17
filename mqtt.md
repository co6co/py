https://www.uppdd.com/info?id=166

上面代码中， options 是客户端连接选项，以下是主要参数说明，其余参数详见https://www.npmjs.com/package/mqtt#connect。
keepalive：心跳时间，默认 60秒，设置 0 为禁用；
clientId： 客户端 ID ，默认通过 'mqttjs_' + Math.random().toString(16).substr(2, 8) 随机生成；
username：连接用户名（可选）；
password：连接密码（可选）；
clean：true，设置为 false 以在离线时接收 QoS 1 和 2 消息；
reconnectPeriod：默认 1000 毫秒，两次重新连接之间的间隔，客户端 ID 重复、认证失败等客户端会重新连接；
connectTimeout：默认 30 * 1000毫秒，收到 CONNACK 之前等待的时间，即连接超时时间；
will：遗嘱消息，当客户端严重断开连接时，Broker 将自动发送的消息。 一般格式为：
topic：要发布的主题
payload：要发布的消息
qos：QoS
retain：保留标志