/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : audit

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2024-07-03 11:23:23
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for sys_user
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_name` varchar(64) DEFAULT NULL COMMENT 'userName',
  `category` int(255) DEFAULT NULL,
  `user_pwd` varchar(256) DEFAULT NULL COMMENT '密码',
  `user_salt` varchar(64) DEFAULT NULL COMMENT 'pwd=盐',
  `user_group_id` int(11) unsigned DEFAULT NULL COMMENT '用户组,仅简单对照:1:超级管理员,2:普通用户',
  `state` int(11) DEFAULT '0' COMMENT '用户状态：0启用，1锁定，2禁用',
  `avatar` varchar(256) DEFAULT NULL COMMENT '图像',
  `remark` varchar(1500) DEFAULT NULL COMMENT '备注',
  `create_time` datetime DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `create_user` bigint(20) DEFAULT NULL,
  `update_time` datetime DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `update_user` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=36 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sys_user
-- ----------------------------
INSERT INTO `sys_user` (`id`,`user_name`,`category`,`user_pwd`,`user_salt`,`user_group_id`,`state`,`avatar`,`remark`,`create_time`,`create_user`,`update_time`,`update_user`) VALUES ('1', 'admin', null, 'f06c1e39faa53c20e2f22d2bbb067a04', 'abcd', '1', '0', null, 'd', '2024-05-09 13:54:18', null, '2024-05-09 13:54:18', '1');

