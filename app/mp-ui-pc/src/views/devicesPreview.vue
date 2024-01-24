<template>
	<div class="container-layout">
		<el-container>
			<el-aside width="200px">
				<device-nav @node-click="onClickNavDevice" ></device-nav>
			</el-aside>
			<el-main>
				<el-scrollbar>
					<div class="content">
						<el-row>
							<el-col :span="19" style="height: 100%">
								<biz-player :player-list="playerList" @selected="onPlayerChecked"></biz-player>
							</el-col>
							<el-col :span="5">
								<div class="content">
									<takler-ptz :current-device-data="currentDeviceData.data"></takler-ptz> 
								</div>
							</el-col>
						</el-row>
					</div>
				</el-scrollbar>
			</el-main>
		</el-container>
	</div>
</template>

<script setup lang="ts" name="basetable">    
	import {   deviceNav } from '../components/devices';
	import { bizPlayer,taklerPtz  } from '../components/biz'; 
	import {playerList,currentDeviceData,onClickNavDevice,onPlayerChecked}   from '../components/devices/preview'; 
	 
</script>
<style scoped lang="less">
	@import '../assets/css/player-split.css';
	@color: #000; // rgba(255, 255, 255, 0.2);
	@bgcolor: #fff; // #464444;
	.el-container {
		height: 82vh;
		overflow: hidden;
		.el-scrollbar {
			background: @bgcolor;
		}

		.el-main {
			padding: 0;
			margin: 0 15px;
			::v-deep .el-scrollbar__view {
				height: 100%;
			}
			.el-row {
				height: 100%;
			}
			.el-scrollbar {
				.playerList {
					.el-icon {
						cursor: pointer;
						font-size: 23px;
						vertical-align: middle;
						padding-bottom: 7px;
						&:hover {
							color: red;
						}
					}
					.video-item {
						position: relative;
						.js {
							position: absolute;
							width: calc(100% - 4px);
							height: calc(100% - 4px);
							left: 2px;
							top: 2px;
						}
					}
				}
				.content {
					padding: 10px;
					.box {
						width: 50%;
						display: inline-block;
						margin-right: 5px;
					}
				}
			}
		}
	}
	.view .title {
		color: var(--el-text-color-regular);
		font-size: 18px;
		margin: 10px 0;
	}
	.view .value {
		color: var(--el-text-color-primary);
		font-size: 16px;
		margin: 10px 0;
	}

	::v-deep .view .radius {
		height: 40px;
		width: 70%;
		border: 1px solid var(--el-border-color);
		border-radius: 0;
		margin-top: 20px;
	}
	::v-deep .el-table tr,
	.el-table__row {
		cursor: pointer;
	}

	.formItem {
		display: flex;
		align-items: center;
		display: inline-block;
		.label {
			display: inline-block;
			color: #aaa;
			padding: 0 5px;
		}
	}

	::v-deep .el-dialog__body {
		height: 70%;
		overflow: auto;
	}
	.menuInfo {
		.el-menu {
			width: auto;
			.el-menu-item {
				padding: 10px;
				height: 40px;
			}
		}
	}
	.el-card__header {
		padding: 1px;
	}
	.el-card__body {
		padding: 5px;
		min-height: 289px;
	}
	.el-card__footer {
		.context {
			text-align: center;
		}
		padding: 5px;
	}

	.el-tree--highlight-current
		.el-tree-node.is-current
		> .el-tree-node__content {
		background: #bdbdc5;
		color: #f56c6c;
	}
</style>
