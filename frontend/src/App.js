import React, { Component } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import {
    Navbar,
    NavbarBrand,
    Container,
} from 'reactstrap';

import { Upload } from './components/Upload';

class App extends Component {
    render() {
        return (
            <div className="App">
                <Navbar color="light" light expand="md">
                    <NavbarBrand href="/">Something About Images</NavbarBrand>
                </Navbar>
                <Container>
                    <Upload />
                </Container>
            </div>
        );
    }
}

export default App;
