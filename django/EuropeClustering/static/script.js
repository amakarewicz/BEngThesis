function GetLabel(id) {
    var labels = document.getElementsByTagName('label');
    for( var i = 0; i < labels.length; i++ ) {
        if (labels[i].htmlFor === id)
        return labels[i];
    }
}

function Hide() {
    var algorithms = document.getElementsByName('algorithm')
    for (var i = 0; i < algorithms.length; i++) {
        if (algorithms[i].matches(':checked')) {
            var selected = algorithms[i]
        }
    }

    if (selected.value === "kmeans") {
        document.getElementById('id_linkage').style.display = 'none';
        document.getElementById('id_eps').style.display = 'none';
        document.getElementById('id_min_samples').style.display = 'none';
        document.getElementById('id_n_clusters').style.display = '';

        GetLabel('id_linkage_0').style.display = 'none';
        GetLabel('id_eps').style.display = 'none';
        GetLabel('id_min_samples').style.display = 'none';
        GetLabel('id_n_clusters').style.display = '';
    } else if (selected.value === "hierarchical") {
        document.getElementById('id_linkage').style.display = '';
        document.getElementById('id_eps').style.display = 'none';
        document.getElementById('id_min_samples').style.display = 'none';
        document.getElementById('id_n_clusters').style.display = '';

        GetLabel('id_linkage_0').style.display = '';
        GetLabel('id_eps').style.display = 'none';
        GetLabel('id_min_samples').style.display = 'none';
        GetLabel('id_n_clusters').style.display = '';
    } else {
        document.getElementById('id_linkage').style.display = 'none';
        document.getElementById('id_eps').style.display = '';
        document.getElementById('id_min_samples').style.display = '';
        document.getElementById('id_n_clusters').style.display = 'none';

        GetLabel('id_linkage_0').style.display = 'none';
        GetLabel('id_eps').style.display = '';
        GetLabel('id_min_samples').style.display = '';
        GetLabel('id_n_clusters').style.display = 'none';
    }
}

window.onload = function() {
    // document.getElementsByClassName('dataframe')[0].thead.style.display = 'table-header-group';
    document.getElementById('id_variables').setAttribute('max-height', '100px');
    document.getElementById('id_variables').setAttribute('overflow-y', 'auto');
    document.getElementById('id_algorithm').onchange = Hide;
    Hide();

    // document.ready(function() {
    //     document.getElementById('id_variables').attr('style', 'height:100px;overflow:scroll;');
    // });
};


