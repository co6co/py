<template>
	<div class="container-layout">
		<el-container>
			<el-header>
				<div class="content">
					<span class="left">
						<em>
						<i class="logo"></i></em>
						<em>
							<h3 class="company">{{ sysInfo.company }}</h3>
							<h3 class="en_company">{{ sysInfo.en_company }}</h3>
						</em>
					</span>
					<span class="right"
						><el-image
							style="width: 80px; height: 80px"
							:src="imageList.current"
							:zoom-rate="1.2"
							:max-scale="7"
							:min-scale="0.2"
							:preview-src-list="imageList.urls"
							:initial-index="4"
							fit="cover"
						/>
					</span>
				</div>
			</el-header>
			<el-main>
				<div class="login-wrap">
					<div class="ms-login">
						<div class="ms-title">
							{{ sysInfo.name }}
							<el-text class="mx-1" size="small" tag="sub">{{
								sysInfo.verson
							}}</el-text>
						</div>
						<el-form
							:model="param"
							:rules="rules"
							ref="login"
							label-width="0px"
							class="ms-content"
						>
							<el-form-item prop="username">
								<el-input v-model="param.username" placeholder="用户名">
									<template #prepend>
										<el-icon>
											<User />
										</el-icon>
									</template>
								</el-input>
							</el-form-item>
							<el-form-item prop="password">
								<el-input
									type="password"
									placeholder="密 码"
									v-model="param.password"
									@keyup.enter="submitForm(login)"
								>
									<template #prepend>
										<el-icon> <Lock /> </el-icon>
									</template>
								</el-input>
							</el-form-item>
							<div class="login-btn">
								<el-button type="primary" @click="submitForm(login)"
									>登录</el-button
								>
							</div>
							<p class="login-tips" v-show="message"> {{ message }}</p>
						</el-form>
					</div>
				</div>
			</el-main>
			<el-footer>
				<div class="content">
					<em class="left">
						<h1><img :src="logo" /></h1>
						<h2>国信协联</h2>
					</em>
					<em class="ritht">
						<span>
							备案序号：&nbsp;<a
								href="https://beian.miit.gov.cn"
								target="_blank"
								>{{ sysInfo.beian?.name }}</a
							>
							<a
								target="_blank"
								:href="beianUrl"
								style="
									display: inline-block;
									text-decoration: none;
									height: 20px;
									line-height: 20px;
								"
								><img :src="ba" style="vertical-align: middle" />
								{{ sysInfo.beian?.serialNumberName }}</a
							>
						</span>
						<span>
							技术支持： &nbsp;<a href="#" target="_blank"
								>江苏惠纬讯信息科技有限公司</a
							>
						</span>
						<span>版权所有(C)2023&nbsp;{{ sysInfo.company }} </span>
					</em>
				</div>
			</el-footer>
		</el-container>
	</div>
</template>

<script setup lang="ts">
	import { ref, reactive } from 'vue';
	import { useTagsStore } from '../store/tags';
	import { usePermissStore } from '../store/permiss';
	import { useRouter } from 'vue-router';
	import { ElMessage } from 'element-plus';
	import type { FormInstance, FormRules } from 'element-plus';
	import { Lock, User } from '@element-plus/icons-vue';
	import { login_svc } from '../api/authen';
	import { json } from 'stream/consumers';
	import { setToken } from '../utils/auth';
	import { Storage } from '../store/Storage';
	import { pkg } from '../utils';
	import logo from '../assets/img/logo2.png';

	import ba from '../assets/img/ba.png';
	import gz from '../assets/img/gz.jpg';
	import { computed } from 'vue'; 

	const imageList=reactive<{current:string,urls:Array<string>}>({current:"",urls:[]})
	imageList.current=gz
	imageList.urls=[gz]

	interface LoginInfo {
		username: string;
		password: string;
	}
	interface BeiAn {
		name: string;
		serialNumber: string;
		serialNumberName: string;
	}
	interface SystemInfo {
		name: string;
		verson: string;
		company: string;
		en_company: string;
		beian?: BeiAn;
	}
	let message = ref('');
	const router = useRouter();
	const param = reactive<LoginInfo>({
		username: '', // "admin",
		password: '', //"admin12345"
	});

	const sysInfo = reactive<SystemInfo>({
		name: '',
		verson: '',
		company: '',
		en_company: '',
	});
	sysInfo.name = pkg.name;
	sysInfo.verson = pkg.version;
	sysInfo.company = pkg.company;
	sysInfo.en_company = pkg.en_company;
	sysInfo.beian = pkg.beian;

	const rules: FormRules = {
		username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
		password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
	};

	let storeage = new Storage();
	const permiss = usePermissStore();
	const login = ref<FormInstance>();
	const beianUrl = computed(() => {
		return (
			'http://www.beian.gov.cn/portal/registerSystemInfo?recordcode=' +
			sysInfo.beian?.serialNumber
		);
	});
	const submitForm = (formEl: FormInstance | undefined) => {
		if (!formEl) return;
		formEl.validate((valid: boolean) => {
			if (valid) {
				login_svc({ userName: param.username, password: param.password })
					.then((res) => {
						message.value = res.message;
						if (res.code == 0) {
							ElMessage.success(message.value);
							setToken(res.data.token, res.data.expireSeconds);
							storeage.set('username', param.username, res.data.expireSeconds);

							const keys =
								permiss.defaultList[
									param.username == 'admin' ? 'admin' : 'user'
								];
							permiss.handleSet(keys);
							storeage.set(
								'ms_keys',
								JSON.stringify(keys),
								res.data.expireSeconds
							);
							router.push('/');
						} else {
							ElMessage.error(message.value);
						}
					})
					.catch((err) => {
						//todo debug  login.vue?t=1697628463669:82 Uncaught (in promise) TypeError: Cannot read properties of undefined (reading 'message')
						message.value = err.message || '请求出错';
						ElMessage.error(err.message);
					});
			} else {
				message.value = '数据验证失败';
				return false;
			}
		});
	};

	const tags = useTagsStore();
	tags.clearTags();
</script>

<style scoped>
	.el-header {
		height: 15vh;
	} 
	.el-header .content {
		overflow: hidden;
		text-align: center;
		padding: 4vh;
		line-height: 34px;
	}
	.el-header .content .left,
	.el-header .content .right {
		display: block;
	}
	.el-header .content .left{float: left;}
	.el-header .content .right{float: right;}
	.el-header .logo {
		width: 133px;
		height: 39px;
		background-image: url(../assets/img/logo.jpg);
		display: inline-block;
	} 
	.el-header em {
		display: inline-block; margin-right: 15px;
	}
	.el-header em h3{text-align: left;}
	.el-header .company {
		font-size: 26px;
		color: #333;
	}
	.el-header .en_company {
		font-size: 12px;
		text-transform: uppercase;
		color: #000;
	}
	.el-main {
		position: relative;
		background-image: url(../assets/img/c.jpg);
		background-size: 100%;
		height: 66vh;
	}
	.el-footer {
		height: 20vh;
		padding: 35px;
		/*background-color: #191919; */
		background-color: #464444;
		
	} 	
	.el-footer .content {
		overflow: hidden;
		text-align: center;
		line-height: 45px; 
	}
	.el-footer .content .left,
	.el-footer .content .ritht {
		display: inline-block;
	}
	.el-footer .content .left {
		border-right: 2px solid #646464; 
		padding-right: 50px;
		margin-right: 50px;
	}
	.el-footer .content .ritht {
		text-align: left;
		font-family: Arial, '微软雅黑', Helvetica, sans-serif;
		font-size: 14px;
		font-style: normal;
		font-weight: normal;
		line-height: 1;
		color: rgba(255, 255, 255, 0.55); 

	}
	.el-footer .content .ritht span {
		display: block;
		line-height: 30px;
	}
	.el-footer .content .ritht span a {
		 color: rgba(255, 255, 255, 0.55);  
	}
	.el-footer .content .left h2 {
		font-size: 32px;
		color: rgba(255, 255, 255, 0.2); 
	 
	} 
	.ms-title {
		width: 100%;
		line-height: 50px;
		text-align: center;
		font-size: 20px;
		color: #fff;
		border-bottom: 1px solid #ddd;
	}

	.ms-login {
		position: absolute;
		left: 50%;
		top: 50%;
		width: 350px;
		margin: -60px 0 0 150px;
		border-radius: 5px;
		background: rgb(90 74 74 / 60%);
		overflow: hidden;
	}
	.ms-content {
		padding: 30px 30px;
	}

	.login-btn {
		text-align: center;
	}

	.login-btn button {
		width: 100%;
		height: 36px;
		margin-bottom: 10px;
	}

	.login-tips {
		font-size: 12px;
		line-height: 30px;
		color: red;
	}
	.company {
		color: #ccc;
		padding: 5px;
		text-align: center;
	}
</style>
 