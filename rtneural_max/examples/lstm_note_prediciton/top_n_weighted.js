outlets = 3

function weighted_random(items, weights) {
    var i;

    for (i = 1; i < weights.length; i++)
        weights[i] += weights[i - 1];
    
    var random = Math.random() * weights[weights.length - 1];
    
    for (i = 0; i < weights.length; i++)
        if (weights[i] > random)
            break;
    
    return items[i];
}

function top_5(weights) {
    return weights
        .map(function(weight, index) { return { weight: weight, index: index }; })
        .sort(function(a, b) { return b.weight - a.weight; })
        .slice(0, 5)
        .map(function(item) { return item.index; });
}

function list()
{
	var weights = arrayfromargs(arguments);
	
	const items = [];
	for (i = 0; i < weights.length; i++)
		items.push(i);
	
	var top = top_5(weights);
	outlet(0,top[0]);
	
	var weighted = weighted_random(items, weights);
		
	outlet(1,weighted);
	outlet(2,top);
}