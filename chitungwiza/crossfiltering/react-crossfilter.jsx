import csv from "./ndx.json";
import React, { useEffect, useState } from "react";
import crossfilter from "crossfilter2";
import * as d3 from "d3";
import { BarChart, PieChart, BubbleChart, LineChart, ChartContext } from "react-dc-js";
const numberFormat = d3.format(".2f");
function App() {
    const [cx, setCx] = useState(null);
    useEffect(() => {
        (async () => {
            const data =  await d3.json('https://raw.githubusercontent.com/ksachikonye/hzevo-server/main/sleep/infraredSleep.json');
            const dateFormatSpecifier = "%Y-%m-%d";
            const dateFormatParser = d3.timeParse(dateFormatSpecifier);
            data.forEach((d) => {
                d.dd = dateFormatParser(d.date);
                d.month = d3.timeMonth(d.dd); // pre-calculate month for better performance
                d.total = +d.total; // coerce to number
                d.awake = +d.awake;
                d.rem = +d.rem;
                d.deep = +d.deep;
                d.light = +d.light;
                d.onset_latency = +d.onset_latency;
                d.breath_average = +d.breath_average;
                d.hr_average = +d.hr_average;
                d.hr_lowest = +d.hr_lowest;
            });
            const cx = crossfilter(data);
            setCx(cx);
        })();
    }, []);
    if (!cx) {
        return <p>Loading Data...</p>;
    }
    const moveMonths = cx.dimension((d) => d.month);
    const volumeByMonthGroup = moveMonths
        .group()
        .reduceSum((d) => d.rem / d.total);
    const gainOrLoss = cx.dimension((d) => (d.awake > d.deep ? "Sufficient" : "Deficient"));
    const gainOrLossGroup = gainOrLoss.group();
    const yearlyDimension = cx.dimension((d) => d3.timeYear(d.dd).getFullYear());
    const monthlyMoveGroup = moveMonths
        .group()
        .reduceSum((d) => Math.abs(d.total - d.awake));
    const yearlyPerformanceGroup = yearlyDimension.group().reduce((p, v) => {
        ++p.count;
        p.absGain += v.light - v.awake;
        p.fluctuation += Math.abs(v.hr_average - v.hr_lowest);
        p.sumIndex += (v.hr_average + v.hr_lowest) / 2;
        p.avgIndex = p.sumIndex / p.count;
        p.percentageGain = p.avgIndex ? (p.absGain / p.avgIndex) * 100 : 0;
        p.fluctuationPercentage = p.avgIndex
            ? (p.fluctuation / p.avgIndex) * 100
            : 0;
        return p;
    }, (p, v) => {
        --p.count;
        p.absGain -= v.light - v.awake;
        p.fluctuation -= Math.abs(v.hr_average - v.hr_lowest);
        p.sumIndex -= (v.hr_average + v.hr_lowest) / 2;
        p.avgIndex = p.count ? p.sumIndex / p.count : 0;
        p.percentageGain = p.avgIndex ? (p.absGain / p.avgIndex) * 100 : 0;
        p.fluctuationPercentage = p.avgIndex
            ? (p.fluctuation / p.avgIndex) * 100
            : 0;
        return p;
    }, () => ({
        count: 0,
        absGain: 0,
        fluctuation: 0,
        fluctuationPercentage: 0,
        sumIndex: 0,
        avgIndex: 0,
        percentageGain: 0
    }));
    const indexAvgByMonthGroup = moveMonths.group().reduce((p, v) => {
        ++p.days;
        p.total += (v.awake + v.onset_latency) / 2;
        p.avg = Math.round(p.total / p.days);
        return p;
    }, (p, v) => {
        --p.days;
        p.total -= (v.awake + v.onset_latency) / 2;
        p.avg = p.days ? Math.round(p.total / p.days) : 0;
        return p;
    }, () => ({ days: 0, total: 0, avg: 0 }));
    return (<div className="App">
      <h1>sdsds</h1>
      <ChartContext>
        <BubbleChart width={990} height={250} transitionDuration={1500} margins={{ top: 10, right: 50, bottom: 30, left: 40 }} dimension={yearlyDimension} group={yearlyPerformanceGroup} colors={d3.schemeRdYlGn[9]} colorDomain={[-500, 500]} colorAccessor={(d) => d.value.absGain} keyAccessor={(p) => p.value.absGain} valueAccessor={(p) => p.value.percentageGain} radiusValueAccessor={(p) => p.value.fluctuationPercentage} maxBubbleRelativeSize={0.3} x={d3.scaleLinear().domain([-2500, 2500])} y={d3.scaleLinear().domain([-100, 100])} r={d3.scaleLinear().domain([0, 4000])} elasticY={true} elasticX={true} yAxisPadding={100} xAxisPadding={500} renderHorizontalGridLines={true} renderVerticalGridLines={true} xAxisLabel={"Index Gain"} yAxisLabel={"Index Gain %"} renderLabel={true} label={(p) => p.key} renderTitle={true} title={(p) => [
        p.key,
        `Index Gain: ${numberFormat(p.value.absGain)}`,
        `Index Gain in Percentage: ${numberFormat(p.value.percentageGain)}%`,
        `Fluctuation / Index Ratio: ${numberFormat(p.value.fluctuationPercentage)}%`
    ].join("\n")}/>
        <PieChart dimension={gainOrLoss} group={gainOrLossGroup} width={180} height={180} radius={80}/>
        <LineChart renderArea={true} width={990} height={200} dimension={moveMonths} group={indexAvgByMonthGroup} x={d3
        .scaleTime()
        .domain([new Date(2021, 0, 1), new Date(2022, 09, 29)])} round={d3.timeMonth.round} xUnits={d3.timeMonths} elasticY={true} renderHorizontalGridLines={true} valueAccessor={(d) => d.value.avg} brushOn={false} stack={{
        group: monthlyMoveGroup,
        name: "yolo",
        accessor: (d) => d.value
    }} rangeChart="testing"/>
        <BarChart id="testing" dimension={moveMonths} group={volumeByMonthGroup} width={990} height={180} radius={80} centerBar={true} gap={1} x={d3
        .scaleTime()
        .domain([new Date(2021, 0, 1), new Date(2022, 09, 29)])} round={d3.timeMonth.round} alwaysUseRounding={true} xUnits={d3.timeMonths}/>
      </ChartContext>
    </div>);
}
export default App;
