import { writeFile } from 'fs/promises';
import path from 'path';
import consola from 'consola';
import { epRoot } from '@co6co/build-utils'; //执行时出现异常 请在 根package.json /devDependencies: "@co6co/build": "workspace:^",
import pkg from '../packages/co6co/package.json'; // need to be checked

function getVersion() {
	const tagVer = process.env.TAG_VERSION;
	if (tagVer) {
		return tagVer.startsWith('v') ? tagVer.slice(1) : tagVer;
	} else {
		return pkg.version;
	}
}

const version = getVersion();

async function main() {
	consola.info(`Version: ${version}`);
	await writeFile(
		path.resolve(epRoot, 'version.ts'),
		`
//auto gen-version from package.json version
export const version = '${version}'
    `
	);
}

main();
