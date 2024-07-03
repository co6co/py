/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : audit

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2024-07-03 11:23:42
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for sys_user_group
-- ----------------------------
DROP TABLE IF EXISTS `sys_user_group`;
CREATE TABLE `sys_user_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `parent_id` int(11) DEFAULT NULL,
  `name` varchar(64) DEFAULT NULL,
  `code` varchar(64) DEFAULT NULL,
  `order` int(255) DEFAULT NULL,
  `create_user` bigint(20) DEFAULT NULL COMMENT '创建人',
  `update_user` bigint(20) DEFAULT NULL COMMENT '修改人',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `group_code` (`code`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sys_user_group
-- ----------------------------
INSERT INTO `sys_user_group` (`id`,`parent_id`,`name`,`code`,`order`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('0', null, '根节点', 'root', null, null, null, '2024-04-29 09:52:58', null);
INSERT INTO `sys_user_group` (`id`,`parent_id`,`name`,`code`,`order`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('1', '0', '超级管理员', 'admin', '1', '1', '1', '2024-05-09 09:35:07', '2024-07-01 13:43:01');
INSERT INTO `sys_user_group` (`id`,`parent_id`,`name`,`code`,`order`,`create_user`,`update_user`,`create_time`,`update_time`) VALUES ('2', '0', '微信用户', 'wx_user', '3', '1', '1', '2024-04-29 09:53:45', '2024-04-29 14:08:34');
