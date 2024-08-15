/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : wx_module_perssion

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2024-08-15 11:50:08
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for sys_config
-- ----------------------------
DROP TABLE IF EXISTS `sys_config`;
CREATE TABLE `sys_config` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) DEFAULT NULL,
  `code` varchar(64) DEFAULT NULL,
  `sys_flag` varchar(1) DEFAULT NULL,
  `dict_flag` varchar(1) DEFAULT NULL COMMENT 'Y:使用字典做配置,N:手动配置',
  `dict_type_id` int(11) DEFAULT NULL COMMENT '字典类型ID',
  `value` varchar(1024) DEFAULT NULL COMMENT '配置值',
  `remark` varchar(2048) DEFAULT NULL COMMENT '备注',
  `create_user` bigint(20) DEFAULT NULL COMMENT '创建人',
  `update_user` bigint(20) DEFAULT NULL COMMENT '修改人',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `code` (`code`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sys_config
-- ----------------------------
INSERT INTO `sys_config` VALUES ('1', '系统配置', 'sys_c2', 'N', 'Y', '2', 'M', '2', '1', '1', '2024-07-19 11:30:19', '2024-07-19 12:12:09');
INSERT INTO `sys_config` VALUES ('2', '当前地点', 'work', 'Y', 'Y', '3', 'km', '', '1', '1', '2024-07-19 12:02:18', '2024-07-19 13:45:17');
INSERT INTO `sys_config` VALUES ('3', '百度地图KEY', 'MAP_BAIDU_KEY_VALUE', 'Y', 'N', null, 'rGdLXdmGVwjB3zSSgG71zbeUuWcjEgP5', 'WEB百度地图APIKEY', '1', null, '2024-07-22 11:50:18', null);
INSERT INTO `sys_config` VALUES ('4', '高德地图KEY', 'MAP_GAODE_KEY_VALUE', 'Y', 'N', null, '1a309adbc6e1d16cbf2a6b5970a95a79', '', '1', null, '2024-07-22 13:41:59', null);
INSERT INTO `sys_config` VALUES ('5', 'SYSTEM配置', 'HK_API_Config', 'N', 'N', null, '{\"key\": \"21865624\",\"secret\": \"K2Opn90TnKnZJYTlTEAq\",\"host\": \"xh.kmxdjj.net\"}', '调用接口获取', '1', '1', '2024-07-25 11:06:15', '2024-07-25 14:16:16');
INSERT INTO `sys_config` VALUES ('6', '上传目录', 'SYS_CONFIG_UPLOAD_PATH', 'Y', 'N', null, 'D:\\temp', '', '1', null, '2024-08-08 10:20:53', null);
