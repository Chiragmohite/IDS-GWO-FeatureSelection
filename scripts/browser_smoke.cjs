const { chromium } = require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.IDS_CHROMIUM_EXECUTABLE ? {executablePath:process.env.IDS_CHROMIUM_EXECUTABLE} : {}),args:["--no-sandbox","--disable-dev-shm-usage","--disable-gpu","--no-zygote"]});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.IDS_BROWSER_URL || 'http://127.0.0.1:8501');
 await page.getByText('A clearer view of network activity.').waitFor();
 await page.screenshot({path:(process.env.IDS_BROWSER_OUTPUT || '/tmp')+'/ids-desktop.png',fullPage:true});
 for(const name of ['Events & investigations','CSV analysis','Model lab','Setup & methodology']){
  await page.getByText(name,{exact:true}).click();
  await page.waitForTimeout(500);
  if(await page.locator('[data-testid="stException"]').count())throw Error('Streamlit exception: '+name);
 }
 await page.getByText('CSV analysis',{exact:true}).click();
 await page.getByText('From flow data to inspectable predictions.').waitFor();
 await page.getByText('Use bundled benchmark sample',{exact:true}).click();
 console.log('sample checked',await page.getByRole('checkbox',{name:'Use bundled benchmark sample'}).isChecked());
 await page.waitForTimeout(1000);
 await page.getByRole('button',{name:'Analyze flows',exact:true}).click();
 try { await page.getByText(/Analyzed 50 flows/).waitFor(); } catch(e) { console.log((await page.locator('body').innerText()).slice(0,8000)); await page.screenshot({path:(process.env.IDS_BROWSER_OUTPUT || '/tmp')+'/ids-error.png',fullPage:true}); throw e; }
 await page.screenshot({path:(process.env.IDS_BROWSER_OUTPUT || '/tmp')+'/ids-analysis.png',fullPage:true});
 await page.getByRole('button',{name:'Save observations to event history',exact:true}).click();
 await page.getByText(/Saved 50 observations/).waitFor();
 await page.getByText('Overview',{exact:true}).click();
 await page.getByText('Recent 50 of 50 records; grouped by UTC hour.',{exact:true}).waitFor();
 const metricValues=page.locator('[data-testid="stMetricValue"]');
 if((await metricValues.first().innerText()).trim()!=='50')throw Error('Populated Overview count incorrect');
 await page.getByText('Refresh observations every 5 seconds',{exact:true}).click();
 if(!await page.getByRole('checkbox',{name:'Refresh observations every 5 seconds'}).isChecked())throw Error('Refresh did not enable');
 // Two simultaneous browser sessions exercise the populated datetime/fragment path.
 const second=await browser.newPage();
 second.on('pageerror',e=>errors.push(e.message));
 await second.goto(process.env.IDS_BROWSER_URL || 'http://127.0.0.1:8501');
 await second.getByText('Recent 50 of 50 records; grouped by UTC hour.',{exact:true}).waitFor();
 await page.waitForTimeout(6500);
 for(const name of ['Events & investigations','Overview']){
  await page.getByText(name,{exact:true}).click();
  if(name==='Overview')await page.getByText('Recent 50 of 50 records; grouped by UTC hour.',{exact:true}).waitFor();
 }
 for(const tab of [page,second]){
  if(await tab.locator('[data-testid="stException"]').count())throw Error('Populated session raised an exception');
 }
 const health=await page.request.get((process.env.IDS_BROWSER_URL || 'http://127.0.0.1:8501')+'/_stcore/health');
 if(!health.ok()||(await health.text()).trim()!=='ok')throw Error('Server unhealthy after populated sessions');
 await second.close();
 await page.getByText('Research results',{exact:true}).first().click();
 await page.getByText('Baseline and selected subsets',{exact:true}).waitFor();
 const downloadPromise=page.waitForEvent('download');
 await page.getByText('Download all evidence',{exact:true}).click();
 const archive=await downloadPromise;
 if(!archive.suggestedFilename().endsWith('.zip'))throw Error('Missing report archive');
 await page.getByText('Traffic lab',{exact:true}).first().click();
 await page.getByRole('button',{name:'Measure local traffic'}).click();
 await page.getByRole('button',{name:'Save measured flows'}).waitFor();
 await page.getByRole('button',{name:'Save measured flows'}).click();
 await page.getByText('Saved 4 observations.',{exact:true}).waitFor();
 await page.getByText('Recorded PCAP',{exact:true}).click();
 await page.getByText('Use the synthetic teaching capture',{exact:true}).click();
 await page.getByRole('button',{name:'Analyze recorded capture'}).click();
 await page.getByRole('button',{name:'Save measured flows'}).waitFor();
 await page.getByRole('button',{name:'Save measured flows'}).click();
 await page.getByText('Saved 2 observations.',{exact:true}).waitFor();
 await page.getByText('Events & investigations',{exact:true}).click();
 await page.getByRole('button',{name:'Explain this prediction'}).click();
 await page.getByText(/Forest base attack score:/).waitFor();
 if(await page.locator('[data-testid="stException"]').count())throw Error('Research/traffic UI exception');

 await page.setViewportSize({width:390,height:844});
 await page.waitForTimeout(700);
 const collapse=page.getByTestId('stSidebarCollapseButton').getByRole('button');
 const box=await collapse.boundingBox();
 if(box && box.x>=0 && box.x+box.width<=390) await collapse.click();
 await page.waitForTimeout(500);
 await page.emulateMedia({reducedMotion:'reduce'});
 await page.screenshot({path:(process.env.IDS_BROWSER_OUTPUT || '/tmp')+'/ids-mobile.png',fullPage:true});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
 console.log(JSON.stringify({pages:7,analysisRows:50,persistedRows:56,loopbackFlows:4,pcapFlows:2,evidenceDownload:true,explanation:true,simultaneousSessions:2,fragmentRefresh:true,healthStatus:health.status(),pageErrors:errors,mobilePageOverflow:overflow}));
 await browser.close();
 if(errors.length||overflow)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
