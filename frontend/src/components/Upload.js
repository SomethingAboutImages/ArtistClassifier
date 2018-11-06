import React, { Component } from 'react';
import Dropzone from 'react-dropzone';
import { Progress, Form, FormGroup, Label, Input, Col, Alert } from 'reactstrap';
import _ from 'lodash';
import './Upload.css';

const BASE_URL = 'http://localhost:5100';

export class Upload extends Component {

    constructor(props) {
        super(props);

        this.state = {
            model: 'artists',
            files: [],
            results: [],
            imgData: '',
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

        const reader = new FileReader();
        reader.readAsDataURL(files[0]);
        reader.addEventListener('load', () => {
            this.setState({ imgData: reader.result });
        });

        fetch(BASE_URL + `/predict/${this.state.model}`, {
            method: 'POST',
            body: data
        })
        .then(res => res.json())
        .then(body => {
            this.setState({ loading: false });
            if (body.status === 'SUCCESS') {
                this.setState({
                    results: body.response,
                    error: ''
                });
            } else {
                this.setState({
                    error: body.message
                });
            }
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
        input = input.replace(/_/g, ' ');
        let splitStr = input.toLowerCase().split(' ');
        for (var i = 0; i < splitStr.length; i++) {
            splitStr[i] = splitStr[i].charAt(0).toUpperCase() + splitStr[i].substring(1);     
        }
        return splitStr.join(' '); 
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
                <Form className="model-form">
                    <Col sm={{size: 6, offset: 3}}>
                        <FormGroup>
                            <Label for="model-select">Select Model:</Label>
                            <Input
                                type="select"
                                name="select"
                                id="model-select"
                                onChange={e => this.setState({model: e.target.value})}
                            >
                                <option value="artists">Artists</option>
                                <option value="resnet50">ResNet50</option>
                                <option value="picasso">Picasso - Not Picasso</option>
                                <option value="picasso_one">One Epoch Picasso - Not Picasso</option>
                            </Input>
                        </FormGroup>
                    </Col>
                </Form>
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
                <div className={'errors ' + (this.state.error ? '' : 'hidden')}>
                    <Alert color="danger">{this.state.error}</Alert>
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
                    <img className={'img-fluid ' + (this.state.imgData ? '' : 'hidden')}
                        src={this.state.imgData}
                        alt="Preview"
                    />
                </div>
            </section>
        )
    }
}
