/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : audit

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2024-07-03 11:23:50
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for sys_user_group_role
-- ----------------------------
DROP TABLE IF EXISTS `sys_user_group_role`;
CREATE TABLE `sys_user_group_role` (
  `user_group_id` int(11) NOT NULL COMMENT '主键id',
  `role_id` bigint(20) NOT NULL COMMENT '主键id',
  `create_user` bigint(20) DEFAULT NULL COMMENT '创建人', 
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间', 
  PRIMARY KEY (`user_group_id`,`role_id`),
  KEY `role_id` (`role_id`),
  CONSTRAINT `sys_user_group_role_ibfk_1` FOREIGN KEY (`user_group_id`) REFERENCES `sys_user_group` (`id`),
  CONSTRAINT `sys_user_group_role_ibfk_2` FOREIGN KEY (`role_id`) REFERENCES `sys_role` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sys_user_group_role
-- ----------------------------
INSERT INTO `sys_user_group_role` (`user_group_id`,`role_id`,`create_user` ,`create_time` ) VALUES ('1', '1', '1',   '2024-05-14 11:10:09' ); 
