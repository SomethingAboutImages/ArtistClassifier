import React, { Component } from 'react';
import Dropzone from 'react-dropzone';
import { Progress } from 'reactstrap';
import _ from 'lodash';
import './Upload.css';

const BASE_URL = 'http://localhost:5000';

export class Upload extends Component {

    constructor(props) {
        super(props);

        this.state = {
            files: [],
            results: [],
            loading: false
        };
    }

    onDrop(files) {
        this.setState({
            files,
            loading: true
        });

        const data = new FormData();
        data.append('file', files[0]);
        data.append('filename', files[0].name);

        fetch(BASE_URL + '/predict/', {
            method: 'POST',
            body: data
        })
        .then(res => res.json())
        .then(body => {
            this.setState({
                results: body.response,
                loading: false
            });
        })
        .catch(err => {
            this.setState({ loading: false });
        });
    }

    onCancel() {
        this.setState({
            files: [],
            loading: false
        });
    }

    prettyFormat(input) {
        return _.capitalize(input.replace('_', ' '));
    }

    render() {
        const results = this.state.results.map((result, i) => (
            <tr>
                <th scope="row">{i+1}</th>
                <td>{this.prettyFormat(result.label)}</td>
                <td>{(result.value * 100).toFixed(2) + '%'}</td>
            </tr>
        ));

        return (
            <section>
                <div>
                    <Dropzone
                        accept="image/jpeg, image/png"
                        onDrop={this.onDrop.bind(this)}
                        onFileDialogCancel={this.onCancel.bind(this)}
                        className='dropzone'
                        acceptClassName='dropzone-accept'
                    >
                        <h1>Drop image file to classify</h1>
                        <h3>Or, click to select a file</h3>
                    </Dropzone>
                    <h2 className={this.state.loading ? '' : 'hidden'}></h2>
                    <Progress className={this.state.loading ? '' : 'hidden'} animated value={100} />
                </div>
                <div className='results'>
                    <h2>Top 5 Confidence Results:</h2>
                    <table className="table table-hover">
                        <thead>
                            <tr>
                                <th scope="col">#</th>
                                <th scope="col">Label</th>
                                <th scope="col">Confidence Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results}
                        </tbody>
                    </table>
                </div>
            </section>
        )
    }
}